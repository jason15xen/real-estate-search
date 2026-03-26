"""
Deterministic Filter Engine — Uses PostgreSQL indexed queries for
room counts, price, area, and location filtering.
"""

import logging

import asyncpg

from src.models.search import (
    AreaCriterion,
    Criterion,
    LocationCriterion,
    PriceCriterion,
    RoomCountCriterion,
)

logger = logging.getLogger(__name__)


async def apply_hard_filters(
    pool: asyncpg.Pool,
    criteria: list[Criterion],
) -> list[int]:
    """
    Applies deterministic filters via PostgreSQL indexed queries.
    Returns a list of property IDs that pass ALL criteria.
    """
    hard_criteria = [
        c for c in criteria
        if isinstance(c, (RoomCountCriterion, PriceCriterion, AreaCriterion, LocationCriterion))
    ]

    conditions = []
    params = []
    param_idx = 1

    for criterion in hard_criteria:
        if isinstance(criterion, RoomCountCriterion):
            col = _room_type_to_column(criterion.room_type)
            if col:
                if criterion.exact_count is not None:
                    conditions.append(f"{col} = ${param_idx}")
                    params.append(criterion.exact_count)
                    param_idx += 1
                if criterion.min_count is not None:
                    conditions.append(f"{col} >= ${param_idx}")
                    params.append(criterion.min_count)
                    param_idx += 1
                if criterion.max_count is not None:
                    conditions.append(f"{col} <= ${param_idx}")
                    params.append(criterion.max_count)
                    param_idx += 1

        elif isinstance(criterion, PriceCriterion):
            if criterion.min_price is not None:
                conditions.append(f"price_usd >= ${param_idx}")
                params.append(criterion.min_price)
                param_idx += 1
            if criterion.max_price is not None:
                conditions.append(f"price_usd <= ${param_idx}")
                params.append(criterion.max_price)
                param_idx += 1

        elif isinstance(criterion, AreaCriterion):
            if criterion.min_sqft is not None:
                conditions.append(f"area_sqft >= ${param_idx}")
                params.append(criterion.min_sqft)
                param_idx += 1
            if criterion.max_sqft is not None:
                conditions.append(f"area_sqft <= ${param_idx}")
                params.append(criterion.max_sqft)
                param_idx += 1

        elif isinstance(criterion, LocationCriterion):
            if criterion.city:
                conditions.append(f"LOWER(city) = LOWER(${param_idx})")
                params.append(criterion.city)
                param_idx += 1
            if criterion.state:
                conditions.append(f"LOWER(state) = LOWER(${param_idx})")
                params.append(criterion.state)
                param_idx += 1
            if criterion.country:
                conditions.append(f"LOWER(country) = LOWER(${param_idx})")
                params.append(criterion.country)
                param_idx += 1
            if criterion.district:
                conditions.append(f"LOWER(district) = LOWER(${param_idx})")
                params.append(criterion.district)
                param_idx += 1

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    query = f"SELECT id FROM properties WHERE {where_clause}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    property_ids = [row["id"] for row in rows]
    logger.info(f"Hard filter: {len(property_ids)} properties match ({len(conditions)} conditions)")
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
