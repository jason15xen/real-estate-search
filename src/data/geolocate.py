"""Detect the user's search area for location-less queries.

Two detection tiers, called by the orchestrator when a parsed query names no place
and the request carries no map bounds:

  1. locate_by_point — browser-geolocation coordinates -> the region POLYGON that
     contains them (city first, county fallback). Offline, exact, reuses
     regions.geom. None when the point is outside polygon coverage.
  2. locate_by_ip — client IP -> DB-IP City Lite .mmdb (local file, in-process).
     US-only; requires both city and state in the answer. None for private IPs,
     non-US, lookup misses, or a missing database file.

Both return {"city": ..., "state": ...} (or {"county": ..., "state": ...}) shaped
for LocationCriterion injection, and never raise — the caller's tier 3 is the
configured default area (Brevard County). The .mmdb ships out of git; refresh it
monthly with `python -m src.data.update_geoip`.
"""

from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache

import asyncpg

from config.settings import settings
from src.data.us_states import abbrev_state

logger = logging.getLogger(__name__)

try:
    import geoip2.database
except ImportError:  # dependency missing — tier 2 disabled, tiers 1/3 unaffected
    geoip2 = None


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


@lru_cache(maxsize=1)
def _reader():
    """Open the .mmdb once per process; None when unavailable (warned once)."""
    if geoip2 is None:
        logger.warning("geoip2 not installed — IP geolocation disabled")
        return None
    try:
        return geoip2.database.Reader(settings.geoip_db_path)
    except (OSError, ValueError) as e:
        logger.warning(f"GeoIP database unavailable ({e}) — IP geolocation disabled")
        return None


def locate_by_ip(ip: str | None) -> dict | None:
    """{"city", "state"} for a US IP the database can place, else None.
    DB-IP Lite carries the state only as a NAME ('Florida'); abbrev_state
    converts it to the 2-letter code the rest of the pipeline stores."""
    if not ip:
        return None
    try:
        addr = ipaddress.ip_address(ip.strip())
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return None  # docker-internal / dev traffic is unlocatable
        reader = _reader()
        if reader is None:
            return None
        resp = reader.city(str(addr))
        if resp.country.iso_code != "US":
            return None
        city = resp.city.name
        state_name = (
            resp.subdivisions.most_specific.name if resp.subdivisions else None
        )
        state = abbrev_state(state_name) if state_name else None
        if not city or not state or len(state) != 2:
            return None  # half an answer would inject an ambiguous criterion
        return {"city": city, "state": state.upper()}
    except Exception:  # noqa: BLE001 — includes geoip2 AddressNotFoundError
        return None
