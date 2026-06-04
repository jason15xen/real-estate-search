"""Geospatial search: match landmark to school distances (fast), else LLM geocode + PostGIS ST_DWithin."""

import json
import logging

import asyncpg

from config.settings import settings
from src.llm_client import get_async_client
from src.models.search import Criterion, ProximityCriterion

logger = logging.getLogger(__name__)

MILES_TO_METERS = 1609.344


def _is_rating_query(landmark_name: str) -> bool:
    """True if query is about school quality, not a specific school."""
    quality_keywords = ["good school", "great school", "top school", "best school",
                        "highly rated school", "high rated school", "quality school"]
    name_lower = landmark_name.lower()
    return any(kw in name_lower for kw in quality_keywords)


async def _filter_by_school_rating(
    conn: asyncpg.Connection,
    property_ids: list[int],
    max_distance_miles: float,
    min_rating: int = 7,
) -> list[int]:
    """Filter properties that have nearby schools with high ratings."""
    rows = await conn.fetch("""
        SELECT DISTINCT ps.property_id
        FROM property_schools ps
        WHERE ps.property_id = ANY($1)
        AND ps.rating >= $2
        AND ps.distance_miles <= $3
    """, property_ids, min_rating, max_distance_miles)

    result = [row["property_id"] for row in rows]
    logger.info(f"School rating filter (>= {min_rating}, within {max_distance_miles}mi): {len(result)} properties")
    return result


async def _filter_by_school(
    conn: asyncpg.Connection,
    property_ids: list[int],
    landmark_name: str,
    max_distance_miles: float,
) -> list[int] | None:
    """Filter by school distance; None if landmark matches no school."""
    if _is_rating_query(landmark_name):
        return await _filter_by_school_rating(conn, property_ids, max_distance_miles)

    # Fuzzy-match school name via pg_trgm similarity (threshold 0.3)
    rows = await conn.fetch("""
        SELECT DISTINCT ps.property_id
        FROM property_schools ps
        WHERE ps.property_id = ANY($1)
        AND similarity(LOWER(ps.school_name), LOWER($2)) > 0.3
        AND ps.distance_miles <= $3
    """, property_ids, landmark_name, max_distance_miles)

    if not rows:
        # School may exist but no properties within distance
        exists = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM property_schools
                WHERE similarity(LOWER(school_name), LOWER($1)) > 0.3
            )
        """, landmark_name)

        if exists:
            logger.info(f"School '{landmark_name}' found but no properties within {max_distance_miles} miles")
            return []
        else:
            return None  # not a school -> caller falls back to geocoding

    result = [row["property_id"] for row in rows]
    logger.info(f"School filter '{landmark_name}' within {max_distance_miles}mi: {len(result)} properties")
    return result


async def geocode_landmark(landmark_name: str) -> tuple[float, float] | None:
    """Estimate landmark coordinates via Azure OpenAI (fallback for non-school landmarks)."""
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
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        # Reject out-of-range coords so a bad LLM response can't break ST_MakePoint.
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            logger.warning(f"Geocode for '{landmark_name}' out of range: ({lat}, {lon})")
            return None
        return (lat, lon)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to geocode '{landmark_name}': {e}")
        return None


async def apply_proximity_filters(
    pool: asyncpg.Pool,
    property_ids: list[int],
    criteria: list[Criterion],
) -> list[int]:
    """Filter IDs by proximity: try school distances first, else PostGIS geocoding."""
    proximity_criteria = [c for c in criteria if isinstance(c, ProximityCriterion)]
    if not proximity_criteria:
        return property_ids

    if not property_ids:
        return []

    result_ids = property_ids

    async with pool.acquire() as conn:
        for pc in proximity_criteria:
            # Step 1: school distance data (fast)
            school_result = await _filter_by_school(
                conn, result_ids, pc.landmark_name, pc.max_distance_miles
            )

            if school_result is not None:
                result_ids = school_result
                continue

            # Step 2: fall back to LLM geocoding + PostGIS (slow)
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
