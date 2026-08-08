"""Detect the user's search area for location-less queries.

locate_by_point — browser-geolocation coordinates -> the region POLYGON that
contains them (city first, county fallback). Offline, exact, reuses regions.geom.
None when the point is outside polygon coverage; the caller then falls back to
the configured default area (Brevard County). Never raises.
"""

from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)

async def locate_by_point(pool: asyncpg.Pool, lat: float, lng: float) -> dict | None:
    """The region whose polygon contains (lat, lng): city ('0') preferred over
    county ('3'). None for invalid coords or points outside polygon coverage."""
    try:
        if not (-90 <= lat <= 90 and -180 <= lng <= 180) or (lat == 0 and lng == 0):
            return None
        async with pool.acquire() as conn:
            # City polygons OVERLAP in the hand-drawn source (Titusville∩Mims…), so
            # a point can sit inside two. Order: city before county, then SMALLEST
            # polygon (most specific claim), then regionid — fully deterministic.
            row = await conn.fetchrow(
                """
                SELECT regionname, statecode, regiontype
                FROM regions
                WHERE geom IS NOT NULL
                  AND regiontype IN ('0', '3')
                  AND ST_Covers(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography)
                ORDER BY regiontype, ST_Area(geom), regionid
                LIMIT 1
                """,
                lng, lat,
            )
        if row is None:
            return None
        if row["regiontype"] == "3":
            return {"county": row["regionname"], "state": row["statecode"]}
        return {"city": row["regionname"], "state": row["statecode"]}
    except Exception:  # noqa: BLE001 — detection must never break a search
        logger.exception("Point geolocation failed")
        return None
