"""
Search Orchestrator — Coordinates the multi-phase search pipeline.

Pipeline:
  1. LLM Query Parser    → structured criteria (maps synonyms to known feature names)
  2. Hard Filters        → PostgreSQL indexed queries
  3. Proximity Filters   → PostGIS spatial queries
  4. Feature Matching    → PostgreSQL array/text matching on room_instances
  5. Return results
"""

import logging

import asyncpg

from src.models.search import FeatureCriterion
from src.search.filter_engine import apply_hard_filters
from src.search.geo_search import apply_proximity_filters
from src.search.query_parser import parse_query

logger = logging.getLogger(__name__)


async def _match_features(
    pool: asyncpg.Pool,
    property_ids: list[int],
    feature_criteria: list[FeatureCriterion],
) -> list[int]:
    """
    Filters property IDs by feature criteria using PostgreSQL queries.
    Handles positive (must have) and negative (must not have) features,
    with optional room_context filtering.
    """
    if not property_ids or not feature_criteria:
        return property_ids

    result_ids = set(property_ids)

    async with pool.acquire() as conn:
        for fc in feature_criteria:
            keyword = f"%{fc.feature.lower()}%"

            if fc.room_context:
                # Feature must be in specific room type
                rows = await conn.fetch("""
                    SELECT DISTINCT property_id FROM room_instances
                    WHERE property_id = ANY($1)
                    AND room_type = $2
                    AND LOWER(features_text) LIKE $3
                """, list(result_ids), fc.room_context, keyword)
            else:
                # Feature anywhere in property
                rows = await conn.fetch("""
                    SELECT DISTINCT property_id FROM room_instances
                    WHERE property_id = ANY($1)
                    AND LOWER(features_text) LIKE $2
                """, list(result_ids), keyword)

            matched_ids = {row["property_id"] for row in rows}

            if fc.negated:
                # Must NOT have this feature — remove matches
                result_ids = result_ids - matched_ids
            else:
                # Must have this feature — keep only matches
                result_ids = result_ids & matched_ids

    return list(result_ids)


async def _load_properties(pool: asyncpg.Pool, property_ids: list[int]) -> list[dict]:
    """Load full property data from PostgreSQL for the final results."""
    if not property_ids:
        return []

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                p.id, p.guid, p.name, p.street, p.district, p.city, p.state,
                p.postal_code, p.country,
                ST_Y(p.geom::geometry) as latitude,
                ST_X(p.geom::geometry) as longitude,
                p.area_sqft, p.price_usd,
                p.bedroom_count, p.bathroom_count, p.kitchen_count,
                p.living_room_count, p.dining_room_count, p.garage_count
            FROM properties p
            WHERE p.id = ANY($1)
            ORDER BY p.id
        """, property_ids)

        properties = []
        for row in rows:
            prop_id = row["id"]

            # Load room instances
            room_rows = await conn.fetch("""
                SELECT room_type, instance_index, features
                FROM room_instances
                WHERE property_id = $1
                ORDER BY room_type, instance_index
            """, prop_id)

            # Group by room type
            rooms_map: dict[str, list[list[str]]] = {}
            for rr in room_rows:
                rt = rr["room_type"]
                if rt not in rooms_map:
                    rooms_map[rt] = []
                rooms_map[rt].append(list(rr["features"]))

            rooms = []
            for room_type, instances in rooms_map.items():
                rooms.append({
                    "Type": room_type,
                    "Count": len(instances),
                    "Instances": [{"Features": feats} for feats in instances],
                })

            # Load nearby schools
            school_rows = await conn.fetch("""
                SELECT school_name, rating, grades, distance_miles, link
                FROM property_schools
                WHERE property_id = $1
                ORDER BY distance_miles
            """, prop_id)

            schools = [
                {
                    "Name": sr["school_name"],
                    "Rating": sr["rating"],
                    "Grades": sr["grades"],
                    "DistanceMiles": float(sr["distance_miles"]),
                    "Link": sr["link"],
                }
                for sr in school_rows
            ]

            properties.append({
                "Id": row["guid"],
                "Name": row["name"],
                "Address": {
                    "Street": row["street"],
                    "District": row["district"],
                    "City": row["city"],
                    "State": row["state"],
                    "PostalCode": row["postal_code"],
                    "Country": row["country"],
                    "Latitude": row["latitude"],
                    "Longitude": row["longitude"],
                },
                "AreaSqft": row["area_sqft"],
                "PriceUSD": row["price_usd"],
                "Rooms": rooms,
                "Schools": schools,
            })

        return properties


async def search(query: str, pool: asyncpg.Pool) -> dict:
    """
    Executes the full search pipeline using PostgreSQL.
    """
    # Phase 1: Parse query
    logger.info(f"Phase 1: Parsing query: '{query}'")
    parsed_query = await parse_query(query)
    logger.info(f"Parsed {len(parsed_query.criteria)} criteria: {parsed_query.understood_intent}")

    # Phase 2: Hard filters (PostgreSQL indexed)
    logger.info("Phase 2: Hard filters (PostgreSQL)")
    property_ids = await apply_hard_filters(pool, parsed_query.criteria)
    after_hard_filter_count = len(property_ids)

    # Phase 3: Proximity filters (PostGIS)
    logger.info("Phase 3: Proximity filters (PostGIS)")
    property_ids = await apply_proximity_filters(pool, property_ids, parsed_query.criteria)
    after_proximity_count = len(property_ids)

    # Phase 4: Feature matching (PostgreSQL)
    feature_criteria = [
        c for c in parsed_query.criteria if isinstance(c, FeatureCriterion)
    ]
    if feature_criteria and property_ids:
        logger.info("Phase 4: Feature matching (PostgreSQL)")
        property_ids = await _match_features(pool, property_ids, feature_criteria)
    after_feature_count = len(property_ids)

    # Load full property data for results
    results = await _load_properties(pool, property_ids)

    stats = {
        "after_hard_filters": after_hard_filter_count,
        "after_proximity_filters": after_proximity_count,
        "after_feature_match": after_feature_count,
        "final_results": len(results),
    }

    logger.info(f"Pipeline complete: {stats}")

    return {
        "results": results,
        "parsed_query": parsed_query,
        "stats": stats,
    }
