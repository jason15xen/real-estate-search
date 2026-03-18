"""
Search Orchestrator — Coordinates the multi-phase search pipeline.

Pipeline:
  1. LLM Query Parser    → structured criteria from natural language
  2. Hard Filters        → eliminate by room count, price, area, location
  3. Proximity Filters   → eliminate by distance to landmarks
  4. Vector Search       → find candidates with matching features (semantic)
  5. LLM Validator       → verify each candidate truly matches ALL criteria
  6. Return results

This architecture ensures:
  - Vector search handles synonym matching (different words, same meaning)
  - Hard filters + validator ensure NO false positives are returned
"""

import logging

from src.models.property import Property
from src.models.search import FeatureCriterion, ParsedQuery
from src.search.filter_engine import apply_hard_filters
from src.search.geo_search import apply_proximity_filters
from src.search.query_parser import parse_query
from src.search.validator import validate_candidates
from src.search.vector_search import search_by_features

logger = logging.getLogger(__name__)


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

    # Phase 4: Vector search for feature matching (if feature criteria exist)
    has_features = any(
        isinstance(c, FeatureCriterion) for c in parsed_query.criteria
    )

    if has_features and candidates:
        logger.info("Phase 4: Vector search for feature matching")
        candidate_names = [p.Name for p in candidates]
        matched_names = search_by_features(parsed_query.criteria, candidate_names)

        # Reorder candidates by vector search ranking
        name_to_prop = {p.Name: p for p in candidates}
        candidates = [
            name_to_prop[name] for name in matched_names if name in name_to_prop
        ]
        after_vector_count = len(candidates)
    else:
        after_vector_count = len(candidates)

    # Phase 5: LLM validation to eliminate false positives
    if has_features and candidates:
        logger.info("Phase 5: LLM validation")
        candidates = await validate_candidates(candidates, parsed_query)

    final_count = len(candidates)

    stats = {
        "total_properties": len(all_properties),
        "after_hard_filters": after_hard_filter_count,
        "after_proximity_filters": after_proximity_count,
        "after_vector_search": after_vector_count,
        "final_results": final_count,
    }

    logger.info(f"Pipeline complete: {stats}")

    return {
        "results": candidates,
        "parsed_query": parsed_query,
        "stats": stats,
    }
