"""
Geospatial Search — Uses school distance data and PostGIS for proximity queries.

Strategy:
  1. Check if the landmark matches a school name in property_schools table (fast, ~5ms)
  2. If no school match, fall back to LLM geocoding + PostGIS ST_DWithin (slow, ~2-3s)
"""

import json
import logging

import asyncpg

from config.settings import settings
from src.llm_client import get_async_client
from src.models.search import Criterion, ProximityCriterion

logger = logging.getLogger(__name__)

MILES_TO_METERS = 1609.344


async def _filter_by_school(
    conn: asyncpg.Connection,
    property_ids: list[int],
    landmark_name: str,
    max_distance_miles: float,
) -> list[int] | None:
    """
    Try to filter by school distance data. Returns filtered IDs,
    or None if the landmark doesn't match any school.
    """
    # Check if any school matches this landmark name (fuzzy match)
    rows = await conn.fetch("""
        SELECT DISTINCT ps.property_id
        FROM property_schools ps
        WHERE ps.property_id = ANY($1)
        AND LOWER(ps.school_name) LIKE $2
        AND ps.distance_miles <= $3
    """, property_ids, f"%{landmark_name.lower()}%", max_distance_miles)

    if not rows:
        # Check if the school exists at all (maybe just no matches within distance)
        exists = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM property_schools
                WHERE LOWER(school_name) LIKE $1
            )
        """, f"%{landmark_name.lower()}%")

        if exists:
            # School exists but no properties within distance
            logger.info(f"School '{landmark_name}' found but no properties within {max_distance_miles} miles")
            return []
        else:
            # Not a school — fall back to geocoding
            return None

    result = [row["property_id"] for row in rows]
    logger.info(f"School filter '{landmark_name}' within {max_distance_miles}mi: {len(result)} properties")
    return result


async def geocode_landmark(landmark_name: str) -> tuple[float, float] | None:
    """
    Uses Azure OpenAI to estimate the coordinates of a named landmark.
    Fallback for non-school landmarks.
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
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to geocode '{landmark_name}': {e}")
        return None


async def apply_proximity_filters(
    pool: asyncpg.Pool,
    property_ids: list[int],
    criteria: list[Criterion],
) -> list[int]:
    """
    Filters property IDs by proximity.
    First tries school distance data (instant), then falls back to PostGIS geocoding.
    """
    proximity_criteria = [c for c in criteria if isinstance(c, ProximityCriterion)]
    if not proximity_criteria:
        return property_ids

    if not property_ids:
        return []

    result_ids = property_ids

    async with pool.acquire() as conn:
        for pc in proximity_criteria:
            # Step 1: Try school distance data (fast)
            school_result = await _filter_by_school(
                conn, result_ids, pc.landmark_name, pc.max_distance_miles
            )

            if school_result is not None:
                # School match found — use it
                result_ids = school_result
                continue

            # Step 2: Fall back to LLM geocoding + PostGIS (slow)
            logger.info(f"'{pc.landmark_name}' not a school, falling back to geocoding")
            if pc.landmark_latitude is None or pc.landmark_longitude is None:
                coords = await geocode_landmark(pc.landmark_name)
                if coords:
                    pc.landmark_latitude, pc.landmark_longitude = coords
                else:
                    logger.warning(f"Skipping proximity filter for '{pc.landmark_name}'")
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
