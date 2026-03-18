"""
LLM Validator — The critical piece that solves the vector search false-positive problem.

After vector search returns candidates (properties with semantically similar features),
this module uses Claude to verify each candidate ACTUALLY meets ALL criteria.

This is what makes the system different from a pure vector search:
- Vector search: "fireplace" → returns properties SIMILAR to "fireplace" (may not have one)
- Our system:    "fireplace" → vector search finds candidates → validator confirms each
                  property truly has a fireplace → only true matches returned.
"""

import asyncio
import json
import logging

from config.settings import settings
from src.llm_client import get_async_client_fast
from src.models.property import Property
from src.models.search import FeatureCriterion, ParsedQuery

logger = logging.getLogger(__name__)

VALIDATION_SYSTEM_PROMPT = """\
You are a strict real estate property validator. Given a property's details and \
a set of search criteria, you must determine whether the property TRULY satisfies \
ALL feature-based criteria.

Rules:
- A feature criterion is satisfied ONLY if the property clearly has that feature.
- Semantic equivalence is allowed: "hardwood floors" satisfies "wood flooring", \
"enclosed pool" satisfies "covered pool".
- But "similar" is NOT enough: having a "garden" does NOT satisfy "pool". \
Having a "standard fireplace" DOES satisfy "fireplace".
- If a feature has a room_context (e.g., feature="fireplace", room_context="Bedroom"), \
the feature must exist in that specific room type.
- You must evaluate EVERY criterion. If even ONE criterion is not met, the property FAILS.

Return ONLY a JSON object:
{
  "passes": true/false,
  "reasoning": "Brief explanation of which criteria passed/failed and why"
}
"""


def _text_match_feature(prop: Property, criterion: FeatureCriterion) -> bool:
    """Fast text-based check: does the property contain this feature keyword?"""
    keyword = criterion.feature.lower()
    if criterion.room_context:
        features = prop.get_features_by_room_type(criterion.room_context)
    else:
        features = prop.get_all_features()
    return any(keyword in f.lower() or f.lower() in keyword for f in features)


def _text_prefilter(
    candidates: list[Property],
    feature_criteria: list[FeatureCriterion],
) -> tuple[list[Property], list[Property]]:
    """
    Split candidates into:
      - direct_matches: all feature keywords found in text → no LLM needed
      - ambiguous: some features missing in text → need LLM for synonym check
    Properties where NO features match at all are dropped immediately.
    """
    direct_matches = []
    ambiguous = []

    for prop in candidates:
        matched = [_text_match_feature(prop, fc) for fc in feature_criteria]
        if all(matched):
            direct_matches.append(prop)
        elif any(matched):
            # Some features found, others might be synonyms → ask LLM
            ambiguous.append(prop)
        # else: no features match at all → drop

    return direct_matches, ambiguous


async def validate_candidates(
    candidates: list[Property],
    parsed_query: ParsedQuery,
) -> list[Property]:
    """
    Two-phase validation:
      1. Fast text matching — properties with exact keyword hits pass instantly
      2. LLM validation — only for ambiguous cases (possible synonyms)
    """
    feature_criteria = [
        c for c in parsed_query.criteria if isinstance(c, FeatureCriterion)
    ]

    if not feature_criteria:
        return candidates

    # Phase A: fast text pre-filter
    direct_matches, ambiguous = _text_prefilter(candidates, feature_criteria)
    logger.info(
        f"Text pre-filter: {len(candidates)} candidates → "
        f"{len(direct_matches)} direct matches, {len(ambiguous)} need LLM"
    )

    if not ambiguous:
        return direct_matches

    # Phase B: LLM validation only for ambiguous candidates
    client = get_async_client_fast()
    ambiguous = ambiguous[: settings.max_candidates_for_validation]

    async def _validate_one(prop: Property) -> Property | None:
        prompt = _build_validation_prompt(prop, feature_criteria)
        try:
            response = await client.messages.create(
                model=settings.anthropic_model_fast,
                max_tokens=300,
                system=VALIDATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                raw = "\n".join(lines)

            result = json.loads(raw)
            passes = result.get("passes", False)
            reasoning = result.get("reasoning", "")

            if passes:
                logger.info(f"✓ {prop.Name} PASSES: {reasoning}")
                return prop
            else:
                logger.info(f"✗ {prop.Name} FAILS: {reasoning}")
                return None

        except Exception as e:
            logger.error(f"Validation error for {prop.Name}: {e}")
            return None

    results = await asyncio.gather(*[_validate_one(p) for p in ambiguous])
    llm_validated = [p for p in results if p is not None]

    validated = direct_matches + llm_validated
    logger.info(
        f"Final: {len(direct_matches)} text-matched + {len(llm_validated)} LLM-validated "
        f"= {len(validated)} total"
    )
    return validated


def _build_validation_prompt(
    prop: Property, feature_criteria: list[FeatureCriterion]
) -> str:
    parts = [
        "## Property Details",
        prop.to_text_description(),
        "",
        "## Criteria to Validate",
    ]
    for i, fc in enumerate(feature_criteria, 1):
        if fc.room_context:
            parts.append(
                f"{i}. Feature '{fc.feature}' must be present in {fc.room_context}"
            )
        else:
            parts.append(f"{i}. Feature '{fc.feature}' must be present in the property")

    parts.append("")
    parts.append("Does this property satisfy ALL of the above criteria?")
    return "\n".join(parts)
