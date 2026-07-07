"""Import points of interest from OpenStreetMap (Overpass API) into the pois table.

Area is derived FROM THE PROPERTIES IN THE DB — one bounding box per county (extent
+ margin). Coverage is tracked in poi_coverage so each county is fetched only once;
the worker calls ensure_coverage() after ingesting new properties, so uploading a
NEW COUNTY auto-imports its POIs in the background (no manual step).

Run a full re-import:  python -m src.data.import_pois

Proximity filtering uses ST_DWithin on pois.geom — the live search path makes NO
external calls; only this import touches Overpass."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
import urllib.request

from src.data.database import get_pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Margin (degrees, ~8 mi) around each county's property extent so a POI just beyond
# the cluster but within a user's search radius is still imported.
MARGIN_DEG = 0.12
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Serializes auto-refreshes so two worker batches can't import the same county at once.
_coverage_lock = asyncio.Lock()

# Postgres advisory-lock key so a manual full re-import (main, a separate process)
# and the worker's auto-refresh can't import concurrently and corrupt each other.
_ADVISORY_LOCK_KEY = 776655

# After a county import fails or returns empty, wait this long before retrying it
# (prevents a retry storm hammering Overpass during an outage). Per-process, in-memory.
COOLDOWN_SECONDS = 900
_attempt_cooldown: dict[str, float] = {}


def _category(tags: dict) -> str | None:
    """Map raw OSM tags to our user-facing POI category, or None to skip."""
    shop, amen, leis = tags.get("shop"), tags.get("amenity"), tags.get("leisure")
    if shop in ("supermarket", "grocery"):
        return "grocery"
    if shop == "convenience":
        return "convenience_store"
    if amen == "place_of_worship":
        return "church"
    if amen == "fuel":
        return "gas_station"
    if amen == "pharmacy":
        return "pharmacy"
    if amen in ("restaurant", "fast_food", "cafe"):
        return "restaurant"
    if amen == "hospital":
        return "hospital"
    if amen == "bank":
        return "bank"
    if amen == "school":
        return "school"
    if leis == "park":
        return "park"
    if leis in ("fitness_centre", "sports_centre"):
        return "gym"
    return None


def _query(bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    box = f"({s},{w},{n},{e})"
    return f"""
[out:json][timeout:180];
(
  nwr["shop"~"^(supermarket|grocery|convenience)$"]{box};
  nwr["amenity"~"^(place_of_worship|fuel|pharmacy|restaurant|fast_food|cafe|hospital|bank|school)$"]{box};
  nwr["leisure"~"^(park|fitness_centre|sports_centre)$"]{box};
);
out center tags;
"""


def _fetch(bbox: tuple[float, float, float, float]) -> dict:
    """POST the Overpass query for one bbox; try mirrors in order. Blocking — call via asyncio.to_thread from async code."""
    data = urllib.parse.urlencode({"data": _query(bbox)}).encode()
    last_err = None
    for url in OVERPASS_URLS:
        try:
            req = urllib.request.Request(
                url, data=data, headers={"User-Agent": "realestatesearch-poi-import/1.0"}
            )
            with urllib.request.urlopen(req, timeout=200) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 — try the next mirror
            logger.warning("Overpass %s failed: %s", url, e)
            last_err = e
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")


def _parse(payload: dict) -> list[tuple]:
    """Overpass JSON -> deduped list of (category, name, lon, lat)."""
    by_osm: dict[tuple, tuple] = {}
    for el in payload.get("elements", []):
        cat = _category(el.get("tags") or {})
        if not cat:
            continue
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:  # way / relation — `out center` gives a centroid
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        by_osm[(el.get("type"), el.get("id"))] = (cat, (el.get("tags") or {}).get("name"), lon, lat)
    return list(by_osm.values())


async def _ensure_tables(conn) -> None:
    """Create/migrate the pois table (with county) + the poi_coverage marker table."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS pois (
            id SERIAL PRIMARY KEY,
            county TEXT,
            category TEXT NOT NULL,
            name TEXT,
            geom GEOGRAPHY(Point, 4326) NOT NULL
        );
    """)
    await conn.execute("ALTER TABLE pois ADD COLUMN IF NOT EXISTS county TEXT;")  # migrate older table
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST(geom);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_pois_category ON pois(category);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_pois_county ON pois(county);")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS poi_coverage (
            county TEXT PRIMARY KEY,
            imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)


async def _county_bboxes(conn, only_uncovered: bool) -> list[tuple[str, tuple, int]]:
    """(county, bbox, property_count) per county; bbox = property extent + margin.
    only_uncovered=True skips counties already in poi_coverage."""
    cond = ("AND COALESCE(NULLIF(county, ''), '(unknown)') NOT IN (SELECT county FROM poi_coverage)"
            if only_uncovered else "")
    rows = await conn.fetch(f"""
        SELECT COALESCE(NULLIF(county, ''), '(unknown)') AS county,
               min(ST_Y(geom::geometry)) AS south, min(ST_X(geom::geometry)) AS west,
               max(ST_Y(geom::geometry)) AS north, max(ST_X(geom::geometry)) AS east,
               count(*) AS n
        FROM properties
        WHERE geom IS NOT NULL {cond}
        GROUP BY 1
    """)
    return [
        (r["county"],
         (r["south"] - MARGIN_DEG, r["west"] - MARGIN_DEG,
          r["north"] + MARGIN_DEG, r["east"] + MARGIN_DEG),
         r["n"])
        for r in rows
    ]


async def _import_county(pool, county: str, bbox: tuple) -> int:
    """Fetch one county's POIs from Overpass, replace that county's rows, mark it covered.
    Returns the number of POIs imported (0 = treated as a soft failure by the caller)."""
    payload = await asyncio.to_thread(_fetch, bbox)  # blocking HTTP off the event loop
    rows = _parse(payload)
    if not rows:
        # Empty-but-successful response (e.g. Overpass overload returning {elements:[]}).
        # Do NOT mark covered or wipe existing rows — leave it for retry, else this county
        # would be permanently stuck with zero POIs and silently match nothing.
        logger.warning("POI import: %s returned 0 POIs — not marking covered (will retry)", county)
        return 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM pois WHERE county = $1", county)
            await conn.executemany(
                "INSERT INTO pois (county, category, name, geom) "
                "VALUES ($1, $2, $3, ST_SetSRID(ST_MakePoint($4, $5), 4326)::geography)",
                [(county, cat, name, lon, lat) for (cat, name, lon, lat) in rows],
            )
            await conn.execute(
                "INSERT INTO poi_coverage (county) VALUES ($1) "
                "ON CONFLICT (county) DO UPDATE SET imported_at = now()",
                county,
            )
    logger.info("POI import: %s -> %d POIs", county, len(rows))
    return len(rows)


async def ensure_coverage(pool) -> int:
    """Import POIs for any county present in `properties` but not yet in poi_coverage.
    Cheap no-op when everything is covered. Returns the number of counties imported.
    Lock-guarded in-process; a Postgres advisory lock guards against a concurrent
    manual full re-import; failed/empty counties back off for COOLDOWN_SECONDS."""
    if _coverage_lock.locked():
        return 0
    async with _coverage_lock:
        async with pool.acquire() as lock_conn:
            if not await lock_conn.fetchval("SELECT pg_try_advisory_lock($1)", _ADVISORY_LOCK_KEY):
                logger.info("POI import running elsewhere — skipping this refresh")
                return 0
            try:
                await _ensure_tables(lock_conn)
                pending = await _county_bboxes(lock_conn, only_uncovered=True)
                imported = 0
                for county, bbox, n in pending:
                    last = _attempt_cooldown.get(county)
                    if last is not None and time.monotonic() - last < COOLDOWN_SECONDS:
                        continue  # backed off after a recent failed/empty attempt
                    logger.info("POI coverage: new county '%s' (%d properties) — importing", county, n)
                    try:
                        got = await _import_county(pool, county, bbox)
                    except Exception as e:  # noqa: BLE001 — one county must not abort the rest
                        logger.warning("POI import failed for %s: %s", county, e)
                        _attempt_cooldown[county] = time.monotonic()
                        continue
                    if got > 0:
                        imported += 1
                        _attempt_cooldown.pop(county, None)
                    else:
                        _attempt_cooldown[county] = time.monotonic()  # empty → back off
                return imported
            finally:
                await lock_conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)


async def main() -> None:
    """Full re-import: refetch every county present in the data. NO upfront TRUNCATE —
    each county is swapped transactionally inside _import_county, so live searches keep
    the previous POIs until the replacement lands, and a mirror outage mid-run can't
    leave the table empty. Holds the advisory lock so it can't race the worker."""
    pool = await get_pool()
    async with pool.acquire() as lock_conn:
        await lock_conn.fetchval("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_KEY)  # wait for any in-flight import
        try:
            await _ensure_tables(lock_conn)
            regions = await _county_bboxes(lock_conn, only_uncovered=False)
            if not regions:
                logger.error("No properties with coordinates — load data first.")
                return
            ok = 0
            for county, bbox, n in regions:
                logger.info("Importing %s (%d properties) bbox=%s", county, n,
                            tuple(round(x, 3) for x in bbox))
                try:
                    if await _import_county(pool, county, bbox) > 0:
                        ok += 1
                except Exception as e:  # noqa: BLE001 — one county must not abort the rest
                    logger.error("Import failed for %s: %s (previous POIs kept)", county, e)
            dist = await lock_conn.fetch(
                "SELECT category, count(*) c FROM pois GROUP BY category ORDER BY c DESC")
        finally:
            await lock_conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
    logger.info("Reimported %d/%d county region(s); %d POIs total:",
                ok, len(regions), sum(r["c"] for r in dist))
    for r in dist:
        logger.info("  %-18s %d", r["category"], r["c"])


if __name__ == "__main__":
    asyncio.run(main())
