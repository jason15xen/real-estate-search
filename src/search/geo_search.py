"""Geospatial search: match landmark to school distances (fast), else LLM geocode + PostGIS ST_DWithin."""

import json
import logging

import asyncpg

from config.settings import settings
from src.llm_client import get_async_client
from src.models.search import AreaRelationCriterion, Criterion, ProximityCriterion

logger = logging.getLogger(__name__)

MILES_TO_METERS = 1609.344

# Default buffers for area-relation queries.
NEAR_BUFFER_MILES = 2.0      # ring added around the area's own extent for "near A"
BETWEEN_BUFFER_MILES = 2.0   # half-width of the corridor for "between A and B"


def _contains_pattern(value: str) -> str:
    """ILIKE 'contains' pattern with LIKE metacharacters escaped (see filter_engine)."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


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
    """Estimate landmark coordinates via the OpenAI model (fallback for non-school landmarks)."""
    client = get_async_client()
    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
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
        raw = (response.choices[0].message.content or "").strip()
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


async def _resolve_area_centroid(
    conn: asyncpg.Connection, place: str
) -> tuple[float, float, float] | None:
    """Resolve a named AREA to (lat, lon, spread_meters) from our own property centroids, falling back to the LLM geocoder (spread 0); None if unresolvable."""
    pat = _contains_pattern(place)
    row = await conn.fetchrow(
        """
        WITH a AS (
            SELECT geom::geometry AS g FROM properties
            WHERE locality ILIKE $1 OR neighborhood ILIKE $1
               OR city ILIKE $1 OR district ILIKE $1
        ), c AS (SELECT ST_Centroid(ST_Collect(g)) AS ctr FROM a)
        SELECT ST_Y(c.ctr) AS lat, ST_X(c.ctr) AS lon,
               (SELECT count(*) FROM a) AS n,
               COALESCE(
                   (SELECT MAX(ST_Distance(a.g::geography, c.ctr::geography)) FROM a),
                   0) AS spread
        FROM c
        """,
        pat,
    )
    if row and row["n"] and row["lat"] is not None:
        return float(row["lat"]), float(row["lon"]), float(row["spread"])

    coords = await geocode_landmark(place)
    if coords:
        return coords[0], coords[1], 0.0
    return None


async def apply_area_relation_filters(
    pool: asyncpg.Pool,
    property_ids: list[int],
    criteria: list[Criterion],
) -> list[int]:
    """Filter IDs by spatial relation to named areas: 'near A' (radius around A, includes A), 'neighbors A' (same radius excluding A), 'between A B' (corridor joining centroids)."""
    rels = [c for c in criteria if isinstance(c, AreaRelationCriterion)]
    if not rels or not property_ids:
        return property_ids

    result_ids = property_ids
    async with pool.acquire() as conn:
        for rc in rels:
            a = await _resolve_area_centroid(conn, rc.place_a)
            if a is None:
                logger.warning("area_relation: could not resolve area '%s'", rc.place_a)
                continue
            lat_a, lon_a, spread_a = a

            if rc.relation == "between" and rc.place_b:
                b = await _resolve_area_centroid(conn, rc.place_b)
                if b is None:
                    logger.warning("area_relation between: could not resolve '%s'", rc.place_b)
                    continue
                lat_b, lon_b, _ = b
                buf = (rc.radius_miles or BETWEEN_BUFFER_MILES) * MILES_TO_METERS
                rows = await conn.fetch(
                    """
                    SELECT id FROM properties
                    WHERE id = ANY($1)
                    AND ST_DWithin(
                        geom,
                        ST_MakeLine(
                            ST_SetSRID(ST_MakePoint($2, $3), 4326),
                            ST_SetSRID(ST_MakePoint($4, $5), 4326)
                        )::geography,
                        $6
                    )
                    """,
                    result_ids, lon_a, lat_a, lon_b, lat_b, buf,
                )
            elif rc.relation == "neighbors":
                # A's neighbours: in the ring around A but NOT in A itself (`IS NOT TRUE` is NULL-safe).
                radius = spread_a + (rc.radius_miles or NEAR_BUFFER_MILES) * MILES_TO_METERS
                rows = await conn.fetch(
                    """
                    SELECT id FROM properties
                    WHERE id = ANY($1)
                    AND ST_DWithin(
                        geom,
                        ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography,
                        $4
                    )
                    AND (
                        locality ILIKE $5 OR neighborhood ILIKE $5
                        OR city ILIKE $5 OR district ILIKE $5
                    ) IS NOT TRUE
                    """,
                    result_ids, lon_a, lat_a, radius, _contains_pattern(rc.place_a),
                )
            else:  # "near" (default) — includes A
                radius = spread_a + (rc.radius_miles or NEAR_BUFFER_MILES) * MILES_TO_METERS
                rows = await conn.fetch(
                    """
                    SELECT id FROM properties
                    WHERE id = ANY($1)
                    AND ST_DWithin(
                        geom,
                        ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography,
                        $4
                    )
                    """,
                    result_ids, lon_a, lat_a, radius,
                )
            result_ids = [r["id"] for r in rows]
            logger.info("area_relation %s '%s': -> %d properties",
                        rc.relation, rc.place_a, len(result_ids))

    return result_ids


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
