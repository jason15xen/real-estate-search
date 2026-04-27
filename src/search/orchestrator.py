"""
Search Orchestrator — Coordinates the multi-phase search pipeline.

Pipeline:
  1. LLM Query Parser    → structured criteria (maps synonyms to known feature names)
  2. Hard Filters        → PostgreSQL indexed queries
  3. Proximity Filters   → PostGIS spatial queries
  4. Feature Matching    → PostgreSQL array/text matching on room_instances
  5. Return results
"""

import asyncio
import logging

import asyncpg

from src.data.feature_registry import registry
from src.models.search import (
    AreaCriterion,
    FeatureCriterion,
    LocationCriterion,
    PriceCriterion,
    PropertyCriterion,
    ProximityCriterion,
    RoomCountCriterion,
)
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
    Match the UNION of a feature and all its alternatives against room_instances
    in a SINGLE SQL query using the GIN-indexed `features` TEXT[] column with
    the overlap operator (&&).

    Used identically for positive and negated criteria — guarantees symmetry:
        |with X| + |without X| = |total|
    """
    terms = alternatives if alternatives else [feature]
    if not terms:
        return set()

    if room_context:
        rows = await conn.fetch("""
            SELECT DISTINCT property_id FROM room_instances
            WHERE property_id = ANY($1)
              AND room_type = $2
              AND features && $3::text[]
        """, property_ids, room_context, terms)
    else:
        rows = await conn.fetch("""
            SELECT DISTINCT property_id FROM room_instances
            WHERE property_id = ANY($1)
              AND features && $2::text[]
        """, property_ids, terms)
    return {row["property_id"] for row in rows}


async def _match_features(
    pool: asyncpg.Pool,
    property_ids: list[int],
    feature_criteria: list[FeatureCriterion],
    feature_alternatives: dict[str, list[str]] | None = None,
) -> list[int]:
    """
    Filters property IDs by feature criteria.

    Each criterion's match is queried against the SAME initial input set in
    PARALLEL (separate pool connections via asyncio.gather). Results are then
    combined in Python via set intersection (positive) or subtraction (negated).

    Set-theory equivalence with sequential narrowing:
        Sequential: ((I ∩ M_1) ∩ M_2) - M_3
        Parallel:    (I ∩ M_1_full ∩ M_2_full) - M_3_full
    where M_i_full = match against full input I.
    Both produce identical final sets because intersection/subtraction are
    associative and commutative on sets.

    For each criterion (positive or negated), the SAME feature set (feature +
    its alternatives from the registry) is used. This guarantees:
        |properties matching with X| + |properties matching without X| = |total|
    """
    if not property_ids or not feature_criteria:
        return property_ids

    feature_alternatives = feature_alternatives or {}
    initial_ids = property_ids  # query each criterion against the same input

    async def _run(fc: FeatureCriterion) -> set[int]:
        alts = feature_alternatives.get(fc.feature, [fc.feature])
        async with pool.acquire() as conn:
            return await _match_feature_set(
                conn, initial_ids, fc.feature, alts, fc.room_context
            )

    matches = await asyncio.gather(*[_run(fc) for fc in feature_criteria])

    result_ids = set(property_ids)
    for fc, matched in zip(feature_criteria, matches):
        alts = feature_alternatives.get(fc.feature, [fc.feature])
        if fc.negated:
            logger.info(
                f"NEGATED '{fc.feature}' (alts={len(alts)}) excluded {len(matched)} properties"
            )
            result_ids = result_ids - matched
        else:
            logger.info(
                f"POSITIVE '{fc.feature}' (alts={len(alts)}) matched {len(matched)} properties"
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


async def _load_guids(pool: asyncpg.Pool, property_ids: list[int]) -> list[str]:
    """Load only the GUIDs for matched property IDs — a single SELECT query.

    /search returns only GUIDs to clients, so loading full property data
    (rooms, schools, etc.) just to extract the GUID is wasted work. This
    lightweight loader replaces the previous N+1 _load_properties for the
    /search hot path.
    """
    if not property_ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT guid FROM properties WHERE id = ANY($1) ORDER BY id",
            property_ids,
        )
    return [row["guid"] for row in rows]


def _criterion_labels(criterion) -> list[str]:
    """Short human-readable labels for each distinct SQL condition a criterion produces."""
    labels: list[str] = []
    if isinstance(criterion, RoomCountCriterion):
        if criterion.exact_count is not None:
            labels.append(f"{criterion.room_type}=={criterion.exact_count}")
        if criterion.min_count is not None:
            labels.append(f"{criterion.room_type}>={criterion.min_count}")
        if criterion.max_count is not None:
            labels.append(f"{criterion.room_type}<={criterion.max_count}")
    elif isinstance(criterion, PriceCriterion):
        if criterion.min_price is not None:
            labels.append(f"price>={criterion.min_price}")
        if criterion.max_price is not None:
            labels.append(f"price<={criterion.max_price}")
    elif isinstance(criterion, AreaCriterion):
        if criterion.min_sqft is not None:
            labels.append(f"area>={criterion.min_sqft}")
        if criterion.max_sqft is not None:
            labels.append(f"area<={criterion.max_sqft}")
    elif isinstance(criterion, LocationCriterion):
        for attr in ("city", "state", "country", "district"):
            val = getattr(criterion, attr)
            if val:
                labels.append(f"{attr}={val}")
    elif isinstance(criterion, PropertyCriterion):
        for attr, op in [
            ("home_type", "="), ("min_rent", ">="), ("max_rent", "<="),
            ("min_year_built", ">="), ("max_year_built", "<="),
            ("min_lot_sqft", ">="), ("max_lot_sqft", "<="),
            ("min_stories", ">="), ("max_stories", "<="),
        ]:
            val = getattr(criterion, attr)
            if val is not None:
                labels.append(f"{attr}{op}{val}")
    return labels


async def _collect_hard_filter_steps(
    pool: asyncpg.Pool,
    criteria: list,
    bounds: dict | None,
) -> list[dict]:
    """
    Progressively apply bounds, then each hard filter criterion one at a time,
    recording the count after each step. Used only in debug mode.
    """
    steps: list[dict] = []

    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT count(*) FROM properties")
    steps.append({"step": "total_properties", "count": int(total)})

    applied: list = []
    prev = int(total)

    if bounds:
        count = len(await apply_hard_filters(pool, applied, bounds=bounds))
        steps.append({
            "step": "bounds",
            "count": count,
            "dropped": prev - count,
        })
        prev = count

    hard_types = (RoomCountCriterion, PriceCriterion, AreaCriterion,
                  LocationCriterion, PropertyCriterion)
    for c in criteria:
        if not isinstance(c, hard_types):
            continue
        applied.append(c)
        labels = _criterion_labels(c) or [c.type.value if hasattr(c, "type") else type(c).__name__]
        count = len(await apply_hard_filters(pool, applied, bounds=bounds))
        steps.append({
            "step": ", ".join(labels),
            "count": count,
            "dropped": prev - count,
        })
        prev = count

    return steps


async def search(
    query: str,
    pool: asyncpg.Pool,
    bounds: dict | None = None,
    debug: bool = False,
) -> dict:
    """
    Executes the full search pipeline using PostgreSQL.

    If bounds (with north/south/east/west) is provided, properties outside
    the bounding box are excluded during Phase 2 (hard filters).

    If debug=True, a per-step count breakdown is included in the result.
    """
    # Phase 1: Parse query
    logger.info(f"Phase 1: Parsing query: '{query}'")
    parsed_query = await parse_query(query)
    logger.info(f"Parsed {len(parsed_query.criteria)} criteria: {parsed_query.understood_intent}")

    filter_steps: list[dict] = []

    # Phase 2: Hard filters (PostgreSQL indexed) — includes map bounds if provided
    logger.info(f"Phase 2: Hard filters (PostgreSQL){' + map bounds' if bounds else ''}")
    if debug:
        filter_steps.extend(
            await _collect_hard_filter_steps(pool, parsed_query.criteria, bounds)
        )
    property_ids = await apply_hard_filters(pool, parsed_query.criteria, bounds=bounds)
    after_hard_filter_count = len(property_ids)

    # Phase 3: Proximity filters (PostGIS) — applied one criterion at a time so
    # each proximity step can be recorded for debug.
    logger.info("Phase 3: Proximity filters (PostGIS)")
    proximity_criteria = [c for c in parsed_query.criteria if isinstance(c, ProximityCriterion)]
    if debug and not proximity_criteria:
        filter_steps.append({
            "step": "proximity_skipped",
            "count": after_hard_filter_count,
            "dropped": 0,
        })
    for pc in proximity_criteria:
        before = len(property_ids)
        property_ids = await apply_proximity_filters(pool, property_ids, [pc])
        if debug:
            filter_steps.append({
                "step": f"proximity: {pc.landmark_name} (<= {pc.max_distance_miles}mi)",
                "count": len(property_ids),
                "dropped": before - len(property_ids),
            })
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
        if debug:
            # Apply each feature criterion one at a time to track progressive drops
            current_ids = set(property_ids)
            async with pool.acquire() as conn:
                for fc in feature_criteria:
                    before = len(current_ids)
                    alts = alternatives.get(fc.feature, [fc.feature])
                    matched = await _match_feature_set(
                        conn, list(current_ids), fc.feature, alts, fc.room_context
                    )
                    if fc.negated:
                        current_ids = current_ids - matched
                        op = "NOT"
                    else:
                        current_ids = current_ids & matched
                        op = "HAS"
                    room_ctx = f" in {fc.room_context}" if fc.room_context else ""
                    filter_steps.append({
                        "step": f"feature: {op} '{fc.feature}'{room_ctx} (alts={len(alts)})",
                        "count": len(current_ids),
                        "dropped": before - len(current_ids),
                    })
            property_ids = list(current_ids)
        else:
            property_ids = await _match_features(
                pool, property_ids, feature_criteria, alternatives or None
            )
    after_feature_count = len(property_ids)

    # Replace the LLM's noisy reconstructed_queries with the deterministic
    # alternatives actually used for matching — this makes the debug view honest.
    parsed_query.reconstructed_queries = sorted(
        {alt for alts in alternatives.values() for alt in alts}
    )

    # Load only the GUIDs (lightweight — /search returns only GUIDs)
    guids = await _load_guids(pool, property_ids)

    stats = {
        "after_hard_filters": after_hard_filter_count,
        "after_proximity_filters": after_proximity_count,
        "after_feature_match": after_feature_count,
        "final_results": len(guids),
    }

    logger.info(f"Pipeline complete: {stats}")

    return {
        "guids": guids,
        "parsed_query": parsed_query,
        "stats": stats,
        "filter_steps": filter_steps if debug else None,
    }
