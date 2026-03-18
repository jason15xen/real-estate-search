"""
Geospatial Search — Filters properties by distance to a named landmark.

Uses the Haversine formula to compute distances between lat/lng coordinates.
For landmark geocoding, uses Claude to resolve landmark names to coordinates
(in production, replace with a geocoding API like Google Maps or Nominatim).
"""

import json
import logging
import math

from config.settings import settings
from src.llm_client import get_async_client_fast
from src.models.property import Property
from src.models.search import Criterion, ProximityCriterion

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3958.8


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Returns distance in miles between two lat/lng points."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_MILES * c


async def geocode_landmark(landmark_name: str) -> tuple[float, float] | None:
    """
    Uses Claude to estimate the coordinates of a named landmark.
    In production, replace this with a proper geocoding API.
    """
    client = get_async_client_fast()
    response = await client.messages.create(
        model=settings.anthropic_model_fast,
        max_tokens=200,
        system=(
            "You are a geocoding assistant. Given a landmark or place name, "
            "return ONLY a JSON object with 'latitude' and 'longitude' fields. "
            "If you cannot determine the location, return {\"error\": \"unknown\"}."
        ),
        messages=[{"role": "user", "content": f"Geocode: {landmark_name}"}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines)
    try:
        data = json.loads(raw)
        if "error" in data:
            logger.warning(f"Could not geocode '{landmark_name}': {data['error']}")
            return None
        return (data["latitude"], data["longitude"])
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse geocoding response for '{landmark_name}': {e}")
        return None


async def apply_proximity_filters(
    properties: list[Property],
    criteria: list[Criterion],
) -> list[Property]:
    """
    Filters properties by proximity to landmarks. Properties must be within
    the specified distance of ALL proximity criteria.
    """
    proximity_criteria = [c for c in criteria if isinstance(c, ProximityCriterion)]
    if not proximity_criteria:
        return properties

    # Resolve coordinates for any landmarks that need geocoding
    for pc in proximity_criteria:
        if pc.landmark_latitude is None or pc.landmark_longitude is None:
            coords = await geocode_landmark(pc.landmark_name)
            if coords:
                pc.landmark_latitude, pc.landmark_longitude = coords
            else:
                logger.warning(
                    f"Skipping proximity filter for '{pc.landmark_name}' — "
                    f"could not determine location"
                )

    # Filter properties
    results = []
    for prop in properties:
        passes_all = True
        for pc in proximity_criteria:
            if pc.landmark_latitude is None or pc.landmark_longitude is None:
                continue  # Skip unresolvable landmarks
            dist = haversine_distance(
                prop.Address.Latitude, prop.Address.Longitude,
                pc.landmark_latitude, pc.landmark_longitude,
            )
            if dist > pc.max_distance_miles:
                passes_all = False
                break
        if passes_all:
            results.append(prop)

    logger.info(
        f"Proximity filter: {len(properties)} → {len(results)} properties"
    )
    return results
