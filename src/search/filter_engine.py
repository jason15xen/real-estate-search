"""Deterministic filter engine: PostgreSQL indexed queries for room counts, price, area, location."""

import logging

import asyncpg

from src.models.search import (
    AreaCriterion,
    Criterion,
    LocationCriterion,
    PriceCriterion,
    PropertyCriterion,
    RoomCountCriterion,
)

logger = logging.getLogger(__name__)


def _like_contains(value: str) -> str:
    """Build a case-insensitive 'contains' pattern for ILIKE, escaping the LIKE
    metacharacters (\\, %, _) in user input so a literal '%' or '_' in a place
    name matches literally instead of as a wildcard. ILIKE's default escape
    char is backslash, so no ESCAPE clause is needed."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


async def apply_hard_filters(
    pool: asyncpg.Pool,
    criteria: list[Criterion],
    bounds: dict | None = None,
    filters: dict | None = None,
) -> list[int]:
    """Return property IDs passing ALL criteria.

    bounds: optional north/south/east/west bbox.
    filters: optional per-field overrides; any key here suppresses the matching
    LLM-extracted sub-condition (e.g. filters.price_max replaces PriceCriterion.max_price).
    """
    hard_criteria = [
        c for c in criteria
        if isinstance(c, (RoomCountCriterion, PriceCriterion, AreaCriterion,
                          LocationCriterion, PropertyCriterion))
    ]

    conditions = []
    params = []
    param_idx = 1

    if bounds:
        try:
            south = float(bounds["south"])
            north = float(bounds["north"])
            west = float(bounds["west"])
            east = float(bounds["east"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid bounds ignored: {e}")
        else:
            conditions.append(
                f"ST_Covers(ST_MakeEnvelope(${param_idx}, ${param_idx + 1}, "
                f"${param_idx + 2}, ${param_idx + 3}, 4326)::geography, geom)"
            )
            params.extend([west, south, east, north])
            param_idx += 4

    # Apply explicit filters first; track covered fields so LLM criteria skip them.
    covered: set[str] = set()
    if filters:
        if filters.get("price_min") is not None:
            conditions.append(f"price_usd >= ${param_idx}")
            params.append(filters["price_min"])
            param_idx += 1
            covered.add("price_min")
        if filters.get("price_max") is not None:
            conditions.append(f"price_usd <= ${param_idx}")
            params.append(filters["price_max"])
            param_idx += 1
            covered.add("price_max")
        if filters.get("beds_min") is not None:
            conditions.append(f"bedroom_count >= ${param_idx}")
            params.append(filters["beds_min"])
            param_idx += 1
            covered.add("beds_min")
        if filters.get("baths_min") is not None:
            conditions.append(f"bathroom_count >= ${param_idx}")
            params.append(filters["baths_min"])
            param_idx += 1
            covered.add("baths_min")
        if filters.get("sqft_min") is not None:
            conditions.append(f"area_sqft >= ${param_idx}")
            params.append(filters["sqft_min"])
            param_idx += 1
            covered.add("sqft_min")
        if filters.get("sqft_max") is not None:
            conditions.append(f"area_sqft <= ${param_idx}")
            params.append(filters["sqft_max"])
            param_idx += 1
            covered.add("sqft_max")
        if filters.get("year_from") is not None:
            conditions.append(f"year_built >= ${param_idx}")
            params.append(filters["year_from"])
            param_idx += 1
            covered.add("year_from")
        if filters.get("year_to") is not None:
            conditions.append(f"year_built <= ${param_idx}")
            params.append(filters["year_to"])
            param_idx += 1
            covered.add("year_to")
        if filters.get("property_types"):
            conditions.append(f"UPPER(home_type) = ANY(${param_idx}::text[])")
            params.append(filters["property_types"])
            param_idx += 1
            covered.add("property_types")
        if filters.get("financing"):
            conditions.append(f"financing && ${param_idx}::text[]")
            params.append(filters["financing"])
            param_idx += 1
            covered.add("financing")

    # LLM-extracted criteria; skip sub-conditions already covered by filters.
    for criterion in hard_criteria:
        if isinstance(criterion, RoomCountCriterion):
            col = _room_type_to_column(criterion.room_type)
            if col:
                room_min_field = {
                    "bedroom": "beds_min",
                    "bathroom": "baths_min",
                }.get(criterion.room_type.lower())
                if criterion.exact_count is not None:
                    conditions.append(f"{col} = ${param_idx}")
                    params.append(criterion.exact_count)
                    param_idx += 1
                if criterion.min_count is not None and room_min_field not in covered:
                    conditions.append(f"{col} >= ${param_idx}")
                    params.append(criterion.min_count)
                    param_idx += 1
                if criterion.max_count is not None:
                    conditions.append(f"{col} <= ${param_idx}")
                    params.append(criterion.max_count)
                    param_idx += 1

        elif isinstance(criterion, PriceCriterion):
            if criterion.min_price is not None and "price_min" not in covered:
                conditions.append(f"price_usd >= ${param_idx}")
                params.append(criterion.min_price)
                param_idx += 1
            if criterion.max_price is not None and "price_max" not in covered:
                conditions.append(f"price_usd <= ${param_idx}")
                params.append(criterion.max_price)
                param_idx += 1

        elif isinstance(criterion, AreaCriterion):
            if criterion.min_sqft is not None and "sqft_min" not in covered:
                conditions.append(f"area_sqft >= ${param_idx}")
                params.append(criterion.min_sqft)
                param_idx += 1
            if criterion.max_sqft is not None and "sqft_max" not in covered:
                conditions.append(f"area_sqft <= ${param_idx}")
                params.append(criterion.max_sqft)
                param_idx += 1

        elif isinstance(criterion, LocationCriterion):
            # Place names use fuzzy (contains) matching. A user's "city" term is
            # matched broadly across the place columns because Zillow's mailing
            # city, the OSM locality, the neighborhood, and the subdivision often
            # disagree (e.g. "Viera" is a locality/neighborhood but filed under
            # mailing city "Rockledge"). state/country stay exact.
            if criterion.city:
                conditions.append(
                    f"(city ILIKE ${param_idx} OR locality ILIKE ${param_idx} "
                    f"OR neighborhood ILIKE ${param_idx} OR district ILIKE ${param_idx})"
                )
                params.append(_like_contains(criterion.city))
                param_idx += 1
            if criterion.locality:
                conditions.append(f"(locality ILIKE ${param_idx} OR city ILIKE ${param_idx})")
                params.append(_like_contains(criterion.locality))
                param_idx += 1
            if criterion.neighborhood:
                conditions.append(f"neighborhood ILIKE ${param_idx}")
                params.append(_like_contains(criterion.neighborhood))
                param_idx += 1
            if criterion.county:
                conditions.append(f"county ILIKE ${param_idx}")
                params.append(_like_contains(criterion.county))
                param_idx += 1
            if criterion.street:
                conditions.append(f"street ILIKE ${param_idx}")
                params.append(_like_contains(criterion.street))
                param_idx += 1
            if criterion.district:
                conditions.append(f"district ILIKE ${param_idx}")
                params.append(_like_contains(criterion.district))
                param_idx += 1
            if criterion.state:
                conditions.append(f"LOWER(state) = LOWER(${param_idx})")
                params.append(criterion.state)
                param_idx += 1
            if criterion.country:
                conditions.append(f"LOWER(country) = LOWER(${param_idx})")
                params.append(criterion.country)
                param_idx += 1

        elif isinstance(criterion, PropertyCriterion):
            if criterion.home_type and "property_types" not in covered:
                conditions.append(f"UPPER(home_type) = UPPER(${param_idx})")
                params.append(criterion.home_type)
                param_idx += 1
            if criterion.min_rent is not None:
                conditions.append(f"rent_estimate >= ${param_idx}")
                params.append(criterion.min_rent)
                param_idx += 1
            if criterion.max_rent is not None:
                conditions.append(f"rent_estimate <= ${param_idx}")
                params.append(criterion.max_rent)
                param_idx += 1
            if criterion.min_year_built is not None and "year_from" not in covered:
                conditions.append(f"year_built >= ${param_idx}")
                params.append(criterion.min_year_built)
                param_idx += 1
            if criterion.max_year_built is not None and "year_to" not in covered:
                conditions.append(f"year_built <= ${param_idx}")
                params.append(criterion.max_year_built)
                param_idx += 1
            if criterion.min_lot_sqft is not None:
                conditions.append(f"lot_size_sqft >= ${param_idx}")
                params.append(criterion.min_lot_sqft)
                param_idx += 1
            if criterion.max_lot_sqft is not None:
                conditions.append(f"lot_size_sqft <= ${param_idx}")
                params.append(criterion.max_lot_sqft)
                param_idx += 1
            if criterion.min_stories is not None:
                conditions.append(f"stories >= ${param_idx}")
                params.append(criterion.min_stories)
                param_idx += 1
            if criterion.max_stories is not None:
                conditions.append(f"stories <= ${param_idx}")
                params.append(criterion.max_stories)
                param_idx += 1
            # has_pool / has_waterfront deliberately skipped: handled as features so
            # positive + negative results sum to the total set.

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    query = f"SELECT id FROM properties WHERE {where_clause}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    property_ids = [row["id"] for row in rows]
    logger.info(
        f"Hard filter: {len(property_ids)} properties match "
        f"({len(conditions)} conditions, bounds={'yes' if bounds else 'no'}, "
        f"filters={'yes' if filters else 'no'})"
    )
    return property_ids


def _room_type_to_column(room_type: str) -> str | None:
    mapping = {
        "bedroom": "bedroom_count",
        "bathroom": "bathroom_count",
        "kitchen": "kitchen_count",
        "living room": "living_room_count",
        "dining room": "dining_room_count",
        "garage": "garage_count",
    }
    return mapping.get(room_type.lower())
