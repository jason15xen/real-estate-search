"""
Geospatial Search — Uses PostGIS for proximity queries.

Replaces the Python Haversine loop with indexed spatial queries.
"""

import json
import logging

import asyncpg

from config.settings import settings
from src.llm_client import get_async_client
from src.models.search import Criterion, ProximityCriterion

logger = logging.getLogger(__name__)

MILES_TO_METERS = 1609.344


async def geocode_landmark(landmark_name: str) -> tuple[float, float] | None:
    """
    Uses Azure OpenAI to estimate the coordinates of a named landmark.
    In production, replace this with a proper geocoding API.
    """
    client = get_async_client()
    try:
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            max_completion_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a geocoding assistant. Given a landmark or place name, "
                        "return ONLY a JSON object with 'latitude' and 'longitude' fields. "
                        'If you cannot determine the location, return {"error": "unknown"}.'
                    ),
                },
                {"role": "user", "content": f"Geocode: {landmark_name}"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines)
        data = json.loads(raw)
        if "error" in data:
            logger.warning(f"Could not geocode '{landmark_name}': {data['error']}")
            return None
        return (data["latitude"], data["longitude"])
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.error(f"Failed to geocode '{landmark_name}': {e}")
        return None


async def apply_proximity_filters(
    pool: asyncpg.Pool,
    property_ids: list[int],
    criteria: list[Criterion],
) -> list[int]:
    """
    Filters property IDs by proximity using PostGIS ST_DWithin.
    """
    proximity_criteria = [c for c in criteria if isinstance(c, ProximityCriterion)]
    if not proximity_criteria:
        return property_ids

    if not property_ids:
        return []

    # Resolve coordinates for landmarks
    for pc in proximity_criteria:
        if pc.landmark_latitude is None or pc.landmark_longitude is None:
            coords = await geocode_landmark(pc.landmark_name)
            if coords:
                pc.landmark_latitude, pc.landmark_longitude = coords
            else:
                logger.warning(f"Skipping proximity filter for '{pc.landmark_name}'")

    result_ids = property_ids
    async with pool.acquire() as conn:
        for pc in proximity_criteria:
            if pc.landmark_latitude is None or pc.landmark_longitude is None:
                continue

            distance_meters = pc.max_distance_miles * MILES_TO_METERS

            rows = await conn.fetch("""
                SELECT id FROM properties
                WHERE id = ANY($1)
                AND ST_DWithin(
                    geom,
                    ST_MakePoint($2, $3)::geography,
                    $4
                )
            """, result_ids, pc.landmark_longitude, pc.landmark_latitude, distance_meters)

            result_ids = [row["id"] for row in rows]

    logger.info(f"Proximity filter: {len(property_ids)} → {len(result_ids)} properties")
    return result_ids
