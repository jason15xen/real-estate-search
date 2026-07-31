"""Load the region catalog (src/data/region/regions_data.sql) into the regions table,
then load region BOUNDARIES from src/data/region/script.sql when present.

The data file is 72k `INSERT INTO Regions ...` statements exported from the source
system. Executing them one-by-one is slow, so this parses the rows out and bulk-loads
them with COPY. The load is idempotent: it replaces the whole table inside one
transaction (readers never see it empty).

script.sql is a fuller SSMS export (UTF-16) whose RegionBoundaries column holds the
boundary as JSON — either a flat list of {"Lat","Lng"} points (one ring) or a list of
such rings (islands / multi-part towns). Each ring becomes one polygon of a
MULTIPOLYGON in regions.geom, which the search uses for geo-location filtering.

Run:  python -m src.data.import_regions

The table itself is defined in schema/postgresql.sql; it is also created here so an
existing database (initdb already ran) picks it up without a manual migration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from src.data.database import get_pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_FILE = Path(__file__).parent / "region" / "regions_data.sql"
# Regions missing from the main export (e.g. communities whose ids postdate the
# snapshot, like Viera). Same INSERT format; loaded after the main file.
SUPPLEMENT_FILE = Path(__file__).parent / "region" / "regions_supplement.sql"
# SSMS export with RegionBoundaries JSON (UTF-16). Optional; boundaries load
# only for the regions it covers.
BOUNDARY_FILE = Path(__file__).parent / "region" / "script.sql"

# One INSERT per line; string values use SQL escaping ('' for a literal quote).
_ROW_RE = re.compile(
    r"VALUES \('([0-9A-Fa-f-]{36})', (\d+), '(\d)', '((?:[^']|'')*)', '([^']*)', '((?:[^']|'')*)'\);"
)

# script.sql statement prefix: VALUES (N'<guid>', <regionid>, <type>, N'<name>', ...
_BOUNDARY_ROW_RE = re.compile(r"VALUES \(N'[0-9a-fA-F-]{36}', (\d+), \d, ")
# ... , NULL, N'[ ...boundary json... ]', <centerlat|NULL>, <centerlng|NULL>)
_BOUNDARY_JSON_RE = re.compile(
    r"NULL, N'(\[.*\])', (?:NULL|[0-9.\-]+), (?:NULL|[0-9.\-]+)\)\s*$", re.S
)

_DDL = """
CREATE TABLE IF NOT EXISTS regions (
    id          UUID PRIMARY KEY,
    regionid    INTEGER NOT NULL UNIQUE,
    regiontype  TEXT NOT NULL,
    regionname  TEXT NOT NULL,
    statecode   TEXT NOT NULL,
    city        TEXT NOT NULL,
    geom        GEOGRAPHY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS idx_regions_type_name ON regions (regiontype, lower(regionname));
CREATE INDEX IF NOT EXISTS idx_regions_statecode ON regions (statecode);
CREATE INDEX IF NOT EXISTS idx_regions_geom ON regions USING GIST(geom);
"""


def _unescape(value: str) -> str:
    return value.replace("''", "'")


def parse_rows(text: str) -> list[tuple]:
    """(id, regionid, regiontype, regionname, statecode, city) per data line."""
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        m = _ROW_RE.search(line)
        if not m:
            # Refuse to half-load: a malformed line means the regex (or file format)
            # is wrong, and silently skipping would drop regions nobody would miss
            # until a search fails to resolve.
            raise ValueError(f"Unparseable line {line_no}: {line[:120]}")
        guid, region_id, rtype, name, state, city = m.groups()
        rows.append((guid, int(region_id), rtype, _unescape(name), state, _unescape(city)))
    return rows


def _rings(boundary) -> list[list[tuple[float, float]]]:
    """Normalize boundary JSON to rings of (lng, lat) — WKT axis order."""
    if not boundary:
        return []
    if isinstance(boundary[0], dict):  # flat list of points: one ring
        boundary = [boundary]
    return [[(p["Lng"], p["Lat"]) for p in ring] for ring in boundary]


def parse_boundaries(path: Path) -> dict[int, str]:
    """regionid -> MULTIPOLYGON WKT from the SSMS export. Rings with <3 distinct
    points are skipped (not a polygon); rings are closed if the source left them
    open. Rows without a parseable boundary are simply absent from the result."""
    wkts: dict[int, str] = {}
    stmts: list[str] = []
    cur: list[str] = []
    with path.open(encoding="utf-16") as f:
        for line in f:
            if line.startswith("INSERT "):
                if cur:
                    stmts.append("".join(cur))
                cur = [line]
            elif cur:
                cur.append(line)
    if cur:
        stmts.append("".join(cur))

    for stmt in stmts:
        head = _BOUNDARY_ROW_RE.search(stmt)
        body = _BOUNDARY_JSON_RE.search(stmt)
        if not head or not body:
            continue
        region_id = int(head.group(1))
        try:
            rings = _rings(json.loads(body.group(1).replace("''", "'")))
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(f"Region {region_id}: unparseable boundary JSON — skipped")
            continue
        polys = []
        for ring in rings:
            if ring and ring[0] != ring[-1]:
                ring = ring + [ring[0]]  # close the ring
            if len(set(ring)) < 3:
                continue  # a point or line, not a polygon
            polys.append("((" + ", ".join(f"{lng} {lat}" for lng, lat in ring) + "))")
        if polys:
            wkts[region_id] = "MULTIPOLYGON(" + ", ".join(polys) + ")"
    return wkts


async def import_boundaries(conn) -> int:
    """Load region polygons from BOUNDARY_FILE into regions.geom. ST_MakeValid
    repairs self-intersections the hand-drawn source rings sometimes contain."""
    if not BOUNDARY_FILE.exists():
        logger.info(f"{BOUNDARY_FILE.name} not present — skipping boundary load")
        return 0
    wkts = parse_boundaries(BOUNDARY_FILE)
    logger.info(f"Parsed {len(wkts)} region boundaries from {BOUNDARY_FILE.name}")
    loaded = 0
    for region_id, wkt in wkts.items():
        try:
            n = await conn.execute(
                """
                UPDATE regions
                SET geom = ST_Multi(ST_MakeValid(ST_GeomFromText($2, 4326)))::geography
                WHERE regionid = $1
                """,
                region_id, wkt,
            )
            loaded += n == "UPDATE 1"
        except Exception as e:  # noqa: BLE001 — one bad polygon must not sink the rest
            logger.warning(f"Region {region_id}: boundary rejected ({e})")
    logger.info(f"Boundaries loaded for {loaded} regions")
    return loaded


async def import_regions() -> int:
    text = DATA_FILE.read_text(encoding="utf-8-sig")  # -sig: file starts with a BOM
    rows = parse_rows(text)
    logger.info(f"Parsed {len(rows)} regions from {DATA_FILE.name}")
    if SUPPLEMENT_FILE.exists():
        extra = parse_rows(SUPPLEMENT_FILE.read_text(encoding="utf-8-sig"))
        rows.extend(extra)
        logger.info(f"Parsed {len(extra)} supplemental regions from {SUPPLEMENT_FILE.name}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(_DDL)
        async with conn.transaction():
            await conn.execute("TRUNCATE regions")
            await conn.copy_records_to_table(
                "regions",
                records=rows,
                columns=["id", "regionid", "regiontype", "regionname", "statecode", "city"],
            )
        count = await conn.fetchval("SELECT count(*) FROM regions")
        await import_boundaries(conn)
    logger.info(f"regions table now holds {count} rows")
    return count


if __name__ == "__main__":
    asyncio.run(import_regions())
