"""Assign region identity (properties.*_region_id) for every property.

Precedence per level (city '0' / county '3' / zipcode '2' / neighborhood '1'):
  1. Zillow's raw region IDs (cityId / countyId / zipcodeId, and for the
     neighborhood level neighborhoodId then parentRegion). Stored UNGUARDED —
     the feed may arrive before the regions table is prepared; a stored id
     becomes searchable when its region row lands. Only parentRegion keeps a
     type check, because it is ambiguous (usually the ZIP region, a
     neighborhood only for unincorporated communities).
  2. Polygon containment — the SMALLEST regions.geom of that type covering the
     property's point — only for records the feed can't place (slim records,
     unknown ids).
  3. ZIP level only: postal_code text -> the type-'2' region row of that name.
  Otherwise NULL — search falls back to name matching for these.

Re-runnable: recomputes every property from scratch (idempotent). Run after
adding/fixing region rows or polygons:

  docker exec realestatesearch-app-1 python -m src.data.backfill_region_ids
"""

from __future__ import annotations

import asyncio
import logging

from src.data.database import close_pool, get_pool

logger = logging.getLogger(__name__)

# One statement per level: Zillow raw ids first, then polygon containment
# applied only where the feed left NULL, then the ZIP text fallback.
_POLYGON_SQL = """
UPDATE properties p SET {col} = sub.regionid
FROM (
    SELECT DISTINCT ON (p2.id) p2.id AS pid, g.regionid
    FROM properties p2
    JOIN regions g ON g.regiontype = $1 AND g.geom IS NOT NULL
        AND ST_Covers(g.geom, p2.geom)
    WHERE p2.{col} IS NULL
    ORDER BY p2.id, ST_Area(g.geom), g.regionid
) sub
WHERE sub.pid = p.id AND p.{col} IS NULL
"""

_ZILLOW_SQL = {
    "city_region_id": """
        UPDATE properties p SET city_region_id = (r.data->>'cityId')::bigint
        FROM raw_properties r
        WHERE r.id = p.guid AND p.city_region_id IS NULL
          AND (r.data->>'cityId') ~ '^[0-9]+$'
    """,
    "county_region_id": """
        UPDATE properties p SET county_region_id = (r.data->>'countyId')::bigint
        FROM raw_properties r
        WHERE r.id = p.guid AND p.county_region_id IS NULL
          AND (r.data->>'countyId') ~ '^[0-9]+$'
    """,
    "zipcode_region_id": """
        UPDATE properties p SET zipcode_region_id = (r.data->>'zipcodeId')::bigint
        FROM raw_properties r
        WHERE r.id = p.guid AND p.zipcode_region_id IS NULL
          AND (r.data->>'zipcodeId') ~ '^[0-9]+$'
    """,
    # neighborhoodId is explicitly a neighborhood -> unguarded. parentRegion is
    # Zillow's "most specific region": a neighborhood for unincorporated
    # communities, a ZIP otherwise — its type guard stays, else ZIP ids would
    # land in the neighborhood column.
    "neighborhood_region_id": """
        UPDATE properties p SET neighborhood_region_id = COALESCE(
            (SELECT (r.data->>'neighborhoodId')::bigint FROM raw_properties r
             WHERE r.id = p.guid AND (r.data->>'neighborhoodId') ~ '^[0-9]+$'),
            (SELECT (r.data->'parentRegion'->>'regionId')::bigint FROM raw_properties r
             WHERE r.id = p.guid AND (r.data->'parentRegion'->>'regionId') ~ '^[0-9]+$'
               AND EXISTS (SELECT 1 FROM regions g WHERE g.regionid = (r.data->'parentRegion'->>'regionId')::bigint AND g.regiontype = '1'))
        )
        WHERE p.neighborhood_region_id IS NULL AND EXISTS (SELECT 1 FROM raw_properties r2 WHERE r2.id = p.guid)
    """,
}

_ZIP_TEXT_SQL = """
    UPDATE properties p SET zipcode_region_id = sub.regionid
    FROM (
        SELECT DISTINCT ON (regionname) regionname, regionid
        FROM regions WHERE regiontype = '2' ORDER BY regionname, regionid
    ) sub
    WHERE p.zipcode_region_id IS NULL AND p.postal_code = sub.regionname
"""

# Feed without cityId: the MAILING CITY name decides, exactly as Zillow would
# assign it — BEFORE polygons, so nested community polygons (Viera inside
# Rockledge/Melbourne) cannot split the city identity between raw-tier and
# polygon-tier records.
_CITY_NAME_SQL = """
    UPDATE properties p SET city_region_id = sub.regionid
    FROM raw_properties r2,
    LATERAL (SELECT g2.regionid FROM regions g2
             WHERE g2.regiontype = '0'
               AND lower(g2.regionname) = lower(trim(r2.data->'address'->>'city'))
               AND g2.statecode = upper(trim(r2.data->'address'->>'state'))
             ORDER BY g2.regionid LIMIT 1) sub
    WHERE r2.id = p.guid AND p.city_region_id IS NULL
"""

# Feed without countyId (MLS): the stated county NAME decides before polygons —
# coastline-hugging county boundaries leave beachfront points metres outside
# ST_Covers.
_COUNTY_NAME_SQL = """
    UPDATE properties p SET county_region_id = sub.regionid
    FROM raw_properties r2,
    LATERAL (SELECT g2.regionid FROM regions g2
             WHERE g2.regiontype = '3'
               AND lower(g2.regionname) = lower(trim(r2.data->>'county'))
               AND g2.statecode = upper(trim(r2.data->'address'->>'state'))
             ORDER BY g2.regionid LIMIT 1) sub
    WHERE r2.id = p.guid AND p.county_region_id IS NULL
"""

_LEVELS = [
    ("city_region_id", "0"),
    ("county_region_id", "3"),
    ("zipcode_region_id", "2"),
    ("neighborhood_region_id", "1"),
]


async def backfill(conn) -> dict[str, int]:
    """Recompute all four region-id columns; returns non-null counts per column."""
    async with conn.transaction():
        await conn.execute(
            "UPDATE properties SET city_region_id=NULL, county_region_id=NULL, "
            "zipcode_region_id=NULL, neighborhood_region_id=NULL"
        )
        for col, rtype in _LEVELS:
            await conn.execute(_ZILLOW_SQL[col])
            if col == "city_region_id":
                await conn.execute(_CITY_NAME_SQL)
            if col == "county_region_id":
                await conn.execute(_COUNTY_NAME_SQL)
            await conn.execute(_POLYGON_SQL.format(col=col), rtype)
        await conn.execute(_ZIP_TEXT_SQL)
    row = await conn.fetchrow(
        "SELECT count(city_region_id) AS city, count(county_region_id) AS county, "
        "count(zipcode_region_id) AS zipcode, count(neighborhood_region_id) AS neighborhood, "
        "count(*) AS total FROM properties"
    )
    return dict(row)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts = await backfill(conn)
    logger.info(f"Region-ID backfill complete: {counts}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
