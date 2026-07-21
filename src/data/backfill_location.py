"""Maintenance backfill: fill EMPTY location fields (county, locality, city, state,
postal_code) of EXISTING properties from Photon reverse geocoding of their coordinates.

Same precedence rule as ingest-time enrichment, enforced in SQL: only empty fields are
written (COALESCE(NULLIF(col, ''), $photon_value)) — existing record data always wins.
Throttled ~1 req/s (photon.reverse_geocode); a coordinate-level cache collapses condo
buildings sharing a point into one call. Re-runnable: each run targets only rows that
still have gaps; failures are skipped and picked up by the next run.

Run:  docker exec realestatesearch-worker-1 python -m src.data.backfill_location [limit]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from src.data.database import get_pool
from src.data.import_pois import ensure_coverage
from src.data.photon import reverse_geocode
from src.data.us_states import abbrev_state

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_MISSING_WHERE = """
    county IS NULL OR county = '' OR locality IS NULL OR locality = ''
    OR city = '' OR state = '' OR postal_code IS NULL OR postal_code = ''
"""


async def main(limit: int | None = None) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT id,
                   round(ST_Y(geom::geometry)::numeric, 5) AS lat,
                   round(ST_X(geom::geometry)::numeric, 5) AS lon
            FROM properties
            WHERE ({_MISSING_WHERE})
              AND NOT (ST_X(geom::geometry) = 0 AND ST_Y(geom::geometry) = 0)
            ORDER BY id
            {f"LIMIT {int(limit)}" if limit else ""}
        """)
    total = len(rows)
    logger.info("Backfill: %d properties with empty location fields", total)
    if not total:
        return

    cache: dict[tuple, dict | None] = {}  # coord -> photon props (None = failed, don't retry this run)
    updated = failed = 0
    started = time.monotonic()
    for i, r in enumerate(rows, 1):
        key = (r["lat"], r["lon"])
        if key in cache:
            props = cache[key]
        else:
            props = await reverse_geocode(float(r["lat"]), float(r["lon"]))
            cache[key] = props
        if not props:
            failed += 1
        else:
            state = abbrev_state(props["state"]) if props.get("state") else None
            async with pool.acquire() as conn:
                # Only empty fields take the geocoded value; record data always wins.
                await conn.execute(
                    """
                    UPDATE properties SET
                        county      = COALESCE(NULLIF(county, ''), $2),
                        locality    = COALESCE(NULLIF(locality, ''), $3),
                        city        = COALESCE(NULLIF(city, ''), $4, city),
                        state       = COALESCE(NULLIF(state, ''), $5, state),
                        postal_code = COALESCE(NULLIF(postal_code, ''), $6),
                        updated_at  = NOW()
                    WHERE id = $1
                    """,
                    r["id"], props.get("county"), props.get("city"),
                    props.get("city"), state, props.get("postcode"),
                )
            updated += 1
        if i % 200 == 0 or i == total:
            rate = i / max(time.monotonic() - started, 1)
            eta_min = (total - i) / rate / 60 if rate else 0
            logger.info("Backfill: %d/%d (%.0f%%) — updated %d, failed %d, cache hits %d, ETA %.0f min",
                        i, total, i * 100 / total, updated, failed, i - len(cache), eta_min)

    logger.info("Backfill done: %d updated, %d failed (re-run to retry failures)", updated, failed)
    # Newly-labeled counties may need POIs for proximity search.
    try:
        n = await ensure_coverage(pool)
        if n:
            logger.info("POI coverage imported for %d new county region(s)", n)
    except Exception as e:  # noqa: BLE001
        logger.warning("POI coverage refresh failed: %s", e)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(lim))
