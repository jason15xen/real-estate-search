"""
Deterministic Filter Engine — Applies hard filters that can be evaluated
without any AI/vector search. These are exact, non-negotiable constraints.

Filters: room counts, price ranges, area ranges, location matching.
"""

import logging

from src.models.property import Property
from src.models.search import (
    AreaCriterion,
    Criterion,
    LocationCriterion,
    PriceCriterion,
    RoomCountCriterion,
)

logger = logging.getLogger(__name__)


def _matches_room_count(prop: Property, criterion: RoomCountCriterion) -> bool:
    count = prop.get_room_count(criterion.room_type)
    if criterion.exact_count is not None and count != criterion.exact_count:
        return False
    if criterion.min_count is not None and count < criterion.min_count:
        return False
    if criterion.max_count is not None and count > criterion.max_count:
        return False
    return True


def _matches_price(prop: Property, criterion: PriceCriterion) -> bool:
    if criterion.min_price is not None and prop.PriceUSD < criterion.min_price:
        return False
    if criterion.max_price is not None and prop.PriceUSD > criterion.max_price:
        return False
    return True


def _matches_area(prop: Property, criterion: AreaCriterion) -> bool:
    if criterion.min_sqft is not None and prop.AreaSqft < criterion.min_sqft:
        return False
    if criterion.max_sqft is not None and prop.AreaSqft > criterion.max_sqft:
        return False
    return True


def _matches_location(prop: Property, criterion: LocationCriterion) -> bool:
    if criterion.city and prop.Address.City.lower() != criterion.city.lower():
        return False
    if criterion.state and prop.Address.State.lower() != criterion.state.lower():
        return False
    if criterion.country and prop.Address.Country.lower() != criterion.country.lower():
        return False
    if criterion.district and prop.Address.District.lower() != criterion.district.lower():
        return False
    return True


def apply_hard_filters(
    properties: list[Property],
    criteria: list[Criterion],
) -> list[Property]:
    """
    Applies all deterministic (non-AI) filters. Returns only properties that
    pass ALL hard filter criteria. Non-hard criteria are ignored here.
    """
    hard_criteria = [
        c for c in criteria
        if isinstance(c, (RoomCountCriterion, PriceCriterion, AreaCriterion, LocationCriterion))
    ]

    if not hard_criteria:
        return properties

    results = []
    for prop in properties:
        passes_all = True
        for criterion in hard_criteria:
            if isinstance(criterion, RoomCountCriterion):
                if not _matches_room_count(prop, criterion):
                    passes_all = False
                    break
            elif isinstance(criterion, PriceCriterion):
                if not _matches_price(prop, criterion):
                    passes_all = False
                    break
            elif isinstance(criterion, AreaCriterion):
                if not _matches_area(prop, criterion):
                    passes_all = False
                    break
            elif isinstance(criterion, LocationCriterion):
                if not _matches_location(prop, criterion):
                    passes_all = False
                    break
        if passes_all:
            results.append(prop)

    logger.info(
        f"Hard filter: {len(properties)} → {len(results)} properties "
        f"({len(hard_criteria)} criteria applied)"
    )
    return results
