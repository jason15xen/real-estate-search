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

from src.data.feature_registry import registry
from src.models.search import FeatureCriterion
from src.search.filter_engine import apply_hard_filters
from src.search.geo_search import apply_proximity_filters
from src.search.query_parser import parse_query

logger = logging.getLogger(__name__)


async def _match_single_feature(
    conn,
    property_ids: list[int],
    feature: str,
    room_context: str | None,
) -> set[int]:
    """Match a single feature keyword against room_instances. Returns matched IDs."""
    keyword = f"%{feature.lower()}%"
    if room_context:
        rows = await conn.fetch("""
            SELECT DISTINCT property_id FROM room_instances
            WHERE property_id = ANY($1)
            AND room_type = $2
            AND LOWER(features_text) LIKE $3
        """, property_ids, room_context, keyword)
    else:
        rows = await conn.fetch("""
            SELECT DISTINCT property_id FROM room_instances
            WHERE property_id = ANY($1)
            AND LOWER(features_text) LIKE $2
        """, property_ids, keyword)
    return {row["property_id"] for row in rows}


async def _match_feature_set(
    conn,
    property_ids: list[int],
    feature: str,
    alternatives: list[str],
    room_context: str | None,
) -> set[int]:
    """
    Match the UNION of a feature and all its alternatives against room_instances.
    Used identically for positive and negated criteria — guarantees symmetry:
        |with X| + |without X| = |total|
    """
    union_ids: set[int] = set()
    terms = alternatives if alternatives else [feature]
    for term in terms:
        matched = await _match_single_feature(conn, property_ids, term, room_context)
        union_ids.update(matched)
    return union_ids


async def _match_features(
    pool: asyncpg.Pool,
    property_ids: list[int],
    feature_criteria: list[FeatureCriterion],
    feature_alternatives: dict[str, list[str]] | None = None,
) -> list[int]:
    """
    Filters property IDs by feature criteria.

    For each criterion (positive or negated), the SAME feature set (feature +
    its alternatives from reconstructed_queries) is used. This guarantees:
        |properties matching with X| + |properties matching without X| = |total|

    Only room_instances.features_text is consulted — description text is NOT
    used, to keep matching strictly tied to "features stored in the database".
    """
    if not property_ids or not feature_criteria:
        return property_ids

    result_ids = set(property_ids)
    feature_alternatives = feature_alternatives or {}

    async with pool.acquire() as conn:
        for fc in feature_criteria:
            id_list = list(result_ids)
            alts = feature_alternatives.get(fc.feature, [fc.feature])
            matched = await _match_feature_set(
                conn, id_list, fc.feature, alts, fc.room_context
            )
            if fc.negated:
                logger.info(
                    f"NEGATED '{fc.feature}' (alts={alts}) excluded {len(matched)} properties"
                )
                result_ids = result_ids - matched
            else:
                logger.info(
                    f"POSITIVE '{fc.feature}' (alts={alts}) matched {len(matched)} properties"
                )
                result_ids = result_ids & matched

    return list(result_ids)


def _build_alternatives(
    feature_criteria: list[FeatureCriterion],
    reconstructed_queries: list[str],  # kept for API compatibility, no longer consulted
) -> dict[str, list[str]]:
    """
    Build a dict mapping each feature criterion's `feature` string → list of
    alternatives, computed DETERMINISTICALLY from the in-memory feature registry.

    The LLM's reconstructed_queries are intentionally ignored here so that
    positive and negated queries get identical alternative sets. This guarantees
        |with X| + |without X| = |total|
    """
    if not feature_criteria:
        return {}

    return {
        fc.feature: registry.get_feature_alternatives(fc.feature)
        for fc in feature_criteria
    }


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


async def search(
    query: str,
    pool: asyncpg.Pool,
    bounds: dict | None = None,
) -> dict:
    """
    Executes the full search pipeline using PostgreSQL.

    If bounds (with north/south/east/west) is provided, properties outside
    the bounding box are excluded during Phase 2 (hard filters).
    """
    # Phase 1: Parse query
    logger.info(f"Phase 1: Parsing query: '{query}'")
    parsed_query = await parse_query(query)
    logger.info(f"Parsed {len(parsed_query.criteria)} criteria: {parsed_query.understood_intent}")

    # Phase 2: Hard filters (PostgreSQL indexed) — includes map bounds if provided
    logger.info(f"Phase 2: Hard filters (PostgreSQL){' + map bounds' if bounds else ''}")
    property_ids = await apply_hard_filters(pool, parsed_query.criteria, bounds=bounds)
    after_hard_filter_count = len(property_ids)

    # Phase 3: Proximity filters (PostGIS)
    logger.info("Phase 3: Proximity filters (PostGIS)")
    property_ids = await apply_proximity_filters(pool, property_ids, parsed_query.criteria)
    after_proximity_count = len(property_ids)

    # Phase 4: Feature matching with alternatives computed from the DB registry
    feature_criteria = [
        c for c in parsed_query.criteria if isinstance(c, FeatureCriterion)
    ]
    alternatives: dict[str, list[str]] = {}
    if feature_criteria and property_ids:
        logger.info("Phase 4: Feature matching (PostgreSQL)")
        alternatives = _build_alternatives(
            feature_criteria, parsed_query.reconstructed_queries
        )
        if alternatives:
            logger.info(f"Feature alternatives: {alternatives}")
        property_ids = await _match_features(
            pool, property_ids, feature_criteria, alternatives or None
        )
    after_feature_count = len(property_ids)

    # Replace the LLM's noisy reconstructed_queries with the deterministic
    # alternatives actually used for matching — this makes the debug view honest.
    parsed_query.reconstructed_queries = sorted(
        {alt for alts in alternatives.values() for alt in alts}
    )

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
