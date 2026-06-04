"""Search orchestrator: parse query -> hard filters -> proximity -> color -> feature matching."""

import asyncio
import logging

import asyncpg

from src.data.feature_registry import registry
from src.models.search import (
    AreaCriterion,
    ColorRoomCriterion,
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


async def _match_feature_set(
    conn,
    property_ids: list[int],
    feature: str,
    alternatives: list[str],
    room_context: str | None,
) -> set[int]:
    """Match feature + alternatives against room_instances via GIN-indexed `features &&`."""
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
    """Filter IDs by feature criteria: match each in parallel, then intersect (positive) / subtract (negated)."""
    if not property_ids or not feature_criteria:
        return property_ids

    feature_alternatives = feature_alternatives or {}
    initial_ids = property_ids

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
    reconstructed_queries: list[str],  # kept for API compat, unused
) -> dict[str, list[str]]:
    """Map each feature -> alternatives from the registry (deterministic; reconstructed_queries ignored)."""
    if not feature_criteria:
        return {}

    return {
        fc.feature: registry.get_feature_alternatives(fc.feature)
        for fc in feature_criteria
    }


async def _match_color_rooms(
    pool: asyncpg.Pool,
    property_ids: list[int],
    color_room_criteria: list[ColorRoomCriterion],
) -> list[int]:
    """Filter IDs by room color: intersect (positive) / subtract (negated) on room_instances.color."""
    if not property_ids or not color_room_criteria:
        return property_ids

    result_ids = set(property_ids)

    async with pool.acquire() as conn:
        for crit in color_room_criteria:
            id_list = list(result_ids)
            rows = await conn.fetch("""
                SELECT DISTINCT property_id FROM room_instances
                WHERE property_id = ANY($1)
                  AND room_type = $2
                  AND color = $3
            """, id_list, crit.room_type, crit.color)
            matched = {row["property_id"] for row in rows}
            if crit.negated:
                logger.info(
                    f"COLOR ROOM NEGATED color={crit.color} room={crit.room_type} "
                    f"excluded {len(matched)} properties"
                )
                result_ids = result_ids - matched
            else:
                logger.info(
                    f"COLOR ROOM POSITIVE color={crit.color} room={crit.room_type} "
                    f"matched {len(matched)} properties"
                )
                result_ids = result_ids & matched

    return list(result_ids)


async def _load_guids(pool: asyncpg.Pool, property_ids: list[int]) -> list[str]:
    """Load GUIDs for matched IDs in one SELECT (/search returns only GUIDs)."""
    if not property_ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT guid FROM properties WHERE id = ANY($1) ORDER BY id",
            property_ids,
        )
    return [row["guid"] for row in rows]


def _criterion_labels(criterion) -> list[str]:
    """Human-readable labels for each SQL condition a criterion produces."""
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


_FILTER_STEP_LABELS = [
    ("price_min", "price>={v}"),
    ("price_max", "price<={v}"),
    ("beds_min", "bedrooms>={v}"),
    ("baths_min", "bathrooms>={v}"),
    ("sqft_min", "area>={v}"),
    ("sqft_max", "area<={v}"),
    ("year_from", "year>={v}"),
    ("year_to", "year<={v}"),
    ("property_types", "home_type∈{v}"),
    ("financing", "financing∋{v}"),
]


async def _collect_hard_filter_steps(
    pool: asyncpg.Pool,
    criteria: list,
    bounds: dict | None,
    filters: dict | None = None,
) -> list[dict]:
    """Debug-only: apply bounds/filters/criteria one at a time, recording count per step."""
    steps: list[dict] = []

    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT count(*) FROM properties")
    steps.append({"step": "total_properties", "count": int(total)})

    applied: list = []
    partial_filters: dict = {}
    prev = int(total)

    if bounds:
        count = len(await apply_hard_filters(pool, applied, bounds=bounds))
        steps.append({
            "step": "bounds",
            "count": count,
            "dropped": prev - count,
        })
        prev = count

    if filters:
        for key, label_tpl in _FILTER_STEP_LABELS:
            value = filters.get(key)
            if value is None or value == [] or value == "":
                continue
            partial_filters[key] = value
            count = len(await apply_hard_filters(
                pool, applied, bounds=bounds, filters=partial_filters
            ))
            steps.append({
                "step": f"filter: {label_tpl.format(v=value)}",
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
        count = len(await apply_hard_filters(
            pool, applied, bounds=bounds, filters=filters
        ))
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
    filters: dict | None = None,
    debug: bool = False,
) -> dict:
    """Run the full search pipeline. bounds restricts to a bbox; filters override LLM
    hard-filter sub-fields; debug adds a per-step count breakdown.
    """
    # Phase 1: Parse
    logger.info(f"Phase 1: Parsing query: '{query}'")
    parsed_query = await parse_query(query)
    logger.info(f"Parsed {len(parsed_query.criteria)} criteria: {parsed_query.understood_intent}")

    filter_steps: list[dict] = []

    # Phase 2: Hard filters (incl. bounds + filters)
    logger.info(
        f"Phase 2: Hard filters (PostgreSQL)"
        f"{' + map bounds' if bounds else ''}"
        f"{' + filters' if filters else ''}"
    )
    if debug:
        filter_steps.extend(
            await _collect_hard_filter_steps(pool, parsed_query.criteria, bounds, filters)
        )
    property_ids = await apply_hard_filters(
        pool, parsed_query.criteria, bounds=bounds, filters=filters
    )
    after_hard_filter_count = len(property_ids)

    # Phase 3: Proximity (one at a time so each step is recorded for debug)
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

    # Phase 3.5: Color-room matching
    color_room_criteria = [
        c for c in parsed_query.criteria if isinstance(c, ColorRoomCriterion)
    ]
    if color_room_criteria and property_ids:
        logger.info("Phase 3.5: Color-room matching (PostgreSQL)")
        if debug:
            current = set(property_ids)
            async with pool.acquire() as conn:
                for crit in color_room_criteria:
                    before = len(current)
                    rows = await conn.fetch("""
                        SELECT DISTINCT property_id FROM room_instances
                        WHERE property_id = ANY($1)
                          AND room_type = $2
                          AND color = $3
                    """, list(current), crit.room_type, crit.color)
                    matched = {row["property_id"] for row in rows}
                    if crit.negated:
                        current = current - matched
                        op = "NOT"
                    else:
                        current = current & matched
                        op = "HAS"
                    filter_steps.append({
                        "step": f"color_room: {op} color={crit.color} in {crit.room_type}",
                        "count": len(current),
                        "dropped": before - len(current),
                    })
            property_ids = list(current)
        else:
            property_ids = await _match_color_rooms(pool, property_ids, color_room_criteria)
    after_color_room_count = len(property_ids)

    # Phase 4: Feature matching (alternatives from registry)
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
        # Sort positives-first, base-before-modifier so debug narration reads logically
        # (result is identical either way since set ops commute).
        feature_criteria = sorted(
            feature_criteria,
            key=lambda fc: (fc.negated, len(fc.feature.split())),
        )
        if debug:
            # Apply one at a time to track progressive drops
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

    # Replace LLM reconstructed_queries with the alternatives actually used (honest debug view)
    parsed_query.reconstructed_queries = sorted(
        {alt for alts in alternatives.values() for alt in alts}
    )

    guids = await _load_guids(pool, property_ids)

    stats = {
        "after_hard_filters": after_hard_filter_count,
        "after_proximity_filters": after_proximity_count,
        "after_color_room_match": after_color_room_count,
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
