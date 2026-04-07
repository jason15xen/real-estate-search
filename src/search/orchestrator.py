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


async def _match_description(
    conn,
    property_ids: list[int],
    feature: str,
) -> set[int]:
    """Match a feature keyword against properties.description. Returns matched IDs."""
    keyword = f"%{feature.lower()}%"
    rows = await conn.fetch("""
        SELECT id FROM properties
        WHERE id = ANY($1)
        AND LOWER(description) LIKE $2
    """, property_ids, keyword)
    return {row["id"] for row in rows}


async def _match_features(
    pool: asyncpg.Pool,
    property_ids: list[int],
    feature_criteria: list[FeatureCriterion],
    alternative_features: list[list[str]] | None = None,
) -> list[int]:
    """
    Filters property IDs by feature criteria.

    If alternative_features is provided (from reconstructed_queries), each
    positive feature is searched using ALL its alternatives (UNION).
    Negated features use exact matching only.

    alternative_features structure:
      [[alt1, alt2, ...], [alt1, alt2, ...], ...]
      One list of alternatives per positive feature criterion, in order.
    """
    if not property_ids or not feature_criteria:
        return property_ids

    result_ids = set(property_ids)
    positive_criteria = [fc for fc in feature_criteria if not fc.negated]
    negated_criteria = [fc for fc in feature_criteria if fc.negated]

    async with pool.acquire() as conn:
        # Process positive features with alternatives (UNION)
        for i, fc in enumerate(positive_criteria):
            id_list = list(result_ids)

            if alternative_features and i < len(alternative_features):
                # Search all alternatives and UNION results
                alts = alternative_features[i]
                union_ids: set[int] = set()
                for alt in alts:
                    matched = await _match_single_feature(
                        conn, id_list, alt, fc.room_context
                    )
                    union_ids.update(matched)
                # Also search description for each alternative
                for alt in alts:
                    desc_matched = await _match_description(conn, id_list, alt)
                    union_ids.update(desc_matched)
                logger.info(
                    f"Feature '{fc.feature}' alternatives={alts} "
                    f"matched {len(union_ids)} properties (UNION incl. description)"
                )
                result_ids = result_ids & union_ids
            else:
                # Standard single-feature match
                matched = await _match_single_feature(
                    conn, id_list, fc.feature, fc.room_context
                )
                # Also search description as fallback
                desc_matched = await _match_description(conn, id_list, fc.feature)
                matched = matched | desc_matched
                result_ids = result_ids & matched

        # Process negated features (exact match, no alternatives)
        # Also exclude from description matches
        for fc in negated_criteria:
            id_list = list(result_ids)
            matched = await _match_single_feature(
                conn, id_list, fc.feature, fc.room_context
            )
            desc_matched = await _match_description(conn, id_list, fc.feature)
            matched = matched | desc_matched
            result_ids = result_ids - matched

    return list(result_ids)


def _build_alternatives(
    feature_criteria: list[FeatureCriterion],
    reconstructed_queries: list[str],
) -> list[list[str]]:
    """
    Build alternative feature lists from reconstructed queries.

    Each reconstructed query is a single feature variant (e.g., "pool",
    "in-ground pool", "private pool"). We group them by matching against
    the original positive feature criteria.

    Returns one list of alternatives per positive feature criterion.
    """
    positive_criteria = [fc for fc in feature_criteria if not fc.negated]
    if not positive_criteria:
        return []

    # Start with each original feature as the only alternative
    alternatives: list[set[str]] = [set() for _ in positive_criteria]

    for rq in reconstructed_queries:
        # Clean the query — strip negated parts and room types
        clean = rq.strip()
        if not clean:
            continue

        # Remove negated parts (anything after " -")
        if " -" in clean:
            clean = clean.split(" -")[0].strip()

        # Try to match this reconstructed query to a positive criterion
        # by checking if the query is related to any original feature
        rq_lower = clean.lower()
        matched_to_criterion = False

        for i, fc in enumerate(positive_criteria):
            fc_lower = fc.feature.lower()
            # Check if they share significant words
            fc_words = set(fc_lower.split())
            rq_words = set(rq_lower.split())
            # Remove common room type words
            room_words = {"kitchen", "bedroom", "bathroom", "living",
                          "room", "dining", "exterior"}
            fc_significant = fc_words - room_words
            rq_significant = rq_words - room_words

            if fc_significant & rq_significant:
                # Shared words — this alternative belongs to this criterion
                alternatives[i].add(clean)
                matched_to_criterion = True
                break

        if not matched_to_criterion:
            # No match found — add to all criteria as a fallback
            # (LLM might have generated a synonym with no shared words)
            for i in range(len(alternatives)):
                alternatives[i].add(clean)

    # Ensure original feature is always included
    result = []
    for i, fc in enumerate(positive_criteria):
        alts = alternatives[i]
        alts.add(fc.feature)  # Always include original
        result.append(sorted(alts))

    return result


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

    # Phase 4: Feature matching with alternatives from reconstructed queries
    feature_criteria = [
        c for c in parsed_query.criteria if isinstance(c, FeatureCriterion)
    ]
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
