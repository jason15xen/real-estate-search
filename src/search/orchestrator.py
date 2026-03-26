"""
Search Orchestrator — Coordinates the multi-phase search pipeline.

Pipeline:
  1. LLM Query Parser    → structured criteria from natural language
                            (maps user words to exact known feature names)
  2. Hard Filters        → eliminate by room count, price, area, location
  3. Proximity Filters   → eliminate by distance to landmarks
  4. Feature Matching    → exact text match on features (no vector search needed)
  5. Return results

Because the query parser maps user input to known feature names at parse time,
vector search and LLM validation are no longer needed.
"""

import logging

from src.models.property import Property
from src.models.search import FeatureCriterion
from src.search.filter_engine import apply_hard_filters
from src.search.geo_search import apply_proximity_filters
from src.search.query_parser import parse_query

logger = logging.getLogger(__name__)


def _matches_feature(prop: Property, criterion: FeatureCriterion) -> bool:
    """Check if a property matches a feature criterion using exact text matching."""
    keyword = criterion.feature.lower()
    if criterion.room_context:
        features = prop.get_features_by_room_type(criterion.room_context)
    else:
        features = prop.get_all_features()
    found = any(keyword in f.lower() or f.lower() in keyword for f in features)
    if criterion.negated:
        return not found
    return found


async def search(
    query: str,
    all_properties: list[Property],
) -> dict:
    """
    Executes the full search pipeline. Returns a dict with:
      - results: list of matching properties
      - parsed_query: the structured interpretation of the query
      - stats: pipeline statistics
    """
    # Phase 1: Parse natural language query into structured criteria
    # (LLM maps user words to exact known feature names)
    logger.info(f"Phase 1: Parsing query: '{query}'")
    parsed_query = await parse_query(query)
    logger.info(f"Parsed {len(parsed_query.criteria)} criteria: {parsed_query.understood_intent}")

    # Phase 2: Apply hard (deterministic) filters
    logger.info("Phase 2: Applying hard filters")
    candidates = apply_hard_filters(all_properties, parsed_query.criteria)
    after_hard_filter_count = len(candidates)

    # Phase 3: Apply proximity filters
    logger.info("Phase 3: Applying proximity filters")
    candidates = await apply_proximity_filters(candidates, parsed_query.criteria)
    after_proximity_count = len(candidates)

    # Phase 4: Apply feature matching (exact text match — no vector search needed)
    feature_criteria = [
        c for c in parsed_query.criteria if isinstance(c, FeatureCriterion)
    ]

    if feature_criteria and candidates:
        logger.info("Phase 4: Feature matching")
        candidates = [
            prop for prop in candidates
            if all(_matches_feature(prop, fc) for fc in feature_criteria)
        ]

    final_count = len(candidates)

    stats = {
        "total_properties": len(all_properties),
        "after_hard_filters": after_hard_filter_count,
        "after_proximity_filters": after_proximity_count,
        "final_results": final_count,
    }

    logger.info(f"Pipeline complete: {stats}")

    return {
        "results": candidates,
        "parsed_query": parsed_query,
        "stats": stats,
    }
