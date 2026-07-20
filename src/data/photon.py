"""Photon (photon.komoot.io) reverse geocoding — ingest-time enrichment of missing
location fields (county, locality) from a property's coordinates.

Used as a FALLBACK only: feeds that carry county/photon data skip this entirely.
Polite to the public instance: global ~1 req/s throttle, short timeout, and any
failure degrades to leaving the field NULL — enrichment must never break ingestion."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from config.settings import settings
from src.data.us_states import abbrev_state
from src.img_analyzer.db_ingest import _extract_locality

logger = logging.getLogger(__name__)

_MIN_INTERVAL_SECONDS = 1.1  # public-instance fair use
_throttle_lock = asyncio.Lock()
_last_call_monotonic = 0.0


async def reverse_geocode(lat: float, lon: float) -> dict | None:
    """Nearest place properties for (lat, lon): {city, county, state, postcode, ...},
    or None on any failure. Globally throttled to ~1 request/second."""
    global _last_call_monotonic
    async with _throttle_lock:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call_monotonic)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_monotonic = time.monotonic()
    try:
        # komoot's public instance 403s generic client UAs — identify ourselves.
        async with httpx.AsyncClient(
            timeout=10, headers={"User-Agent": "realestatesearch-ingest/1.0"}
        ) as client:
            resp = await client.get(
                f"{settings.photon_url}/reverse", params={"lat": lat, "lon": lon}
            )
            resp.raise_for_status()
            features = resp.json().get("features") or []
            if not features:
                return None
            return features[0].get("properties") or {}
    except Exception as e:  # noqa: BLE001 — enrichment is best-effort
        logger.warning("Photon reverse geocode failed for (%s, %s): %s", lat, lon, e)
        return None


async def enrich_location_fields(data: dict) -> bool:
    """Supplement MISSING location fields in a raw Zillow record IN PLACE using
    reverse geocoding: address.city / address.state / address.zipcode, county, and
    the Photon locality structure. Fields the feed already provides are never
    touched, and complete records cost zero network calls. Returns True if
    anything was filled."""
    if not settings.photon_enrich:
        return False
    addr = data.get("address") if isinstance(data.get("address"), dict) else {}
    missing = {
        "county": not str(data.get("county") or "").strip(),
        # "missing" = the ingest extractor would yield NULL (covers absent AND
        # degenerate Photon structures, e.g. empty features)
        "locality": _extract_locality(data) is None,
        "city": not str(addr.get("city") or "").strip(),
        "state": not str(addr.get("state") or "").strip(),
        "zipcode": not str(addr.get("zipcode") or "").strip(),
    }
    if not any(missing.values()):
        return False

    try:
        lat, lon = float(data.get("latitude") or 0), float(data.get("longitude") or 0)
    except (TypeError, ValueError):
        return False
    if not lat or not lon or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False

    props = await reverse_geocode(lat, lon)
    if not props:
        return False

    filled = False
    if missing["county"] and props.get("county"):
        data["county"] = props["county"]
        filled = True
    if missing["locality"] and props.get("city"):
        # Photon's `city` is the OSM place name (e.g. "Viera") — exactly what the
        # locality column exists to hold. Mimic the feed's exact Photon response
        # structure so _extract_locality picks it up unchanged.
        data["PhotonPropertyFullAddress"] = {"features": [{"properties": props}]}
        filled = True
    addr_updates = {}
    if missing["city"] and props.get("city"):
        addr_updates["city"] = props["city"]
    if missing["state"] and props.get("state"):
        addr_updates["state"] = abbrev_state(props["state"])
    if missing["zipcode"] and props.get("postcode"):
        addr_updates["zipcode"] = props["postcode"]
    if addr_updates:
        data["address"] = {**addr, **addr_updates}
        filled = True
    return filled
