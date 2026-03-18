"""
LLM Validator — The critical piece that solves the vector search false-positive problem.

After vector search returns candidates (properties with semantically similar features),
this module uses Claude to verify each candidate ACTUALLY meets ALL criteria.

This is what makes the system different from a pure vector search:
- Vector search: "fireplace" → returns properties SIMILAR to "fireplace" (may not have one)
- Our system:    "fireplace" → vector search finds candidates → validator confirms each
                  property truly has a fireplace → only true matches returned.
"""

import json
import logging

from config.settings import settings
from src.llm_client import get_async_client
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


async def validate_candidates(
    candidates: list[Property],
    parsed_query: ParsedQuery,
) -> list[Property]:
    """
    Validates each candidate property against the feature criteria using Claude.
    Returns only properties that truly match ALL criteria.
    """
    feature_criteria = [
        c for c in parsed_query.criteria if isinstance(c, FeatureCriterion)
    ]

    if not feature_criteria:
        return candidates

    client = get_async_client()
    validated = []

    # Limit candidates to avoid excessive API calls
    candidates_to_check = candidates[: settings.max_candidates_for_validation]

    for prop in candidates_to_check:
        prompt = _build_validation_prompt(prop, feature_criteria)

        try:
            response = await client.messages.create(
                model=settings.anthropic_model,
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
                validated.append(prop)
                logger.info(f"✓ {prop.Name} PASSES validation: {reasoning}")
            else:
                logger.info(f"✗ {prop.Name} FAILS validation: {reasoning}")

        except Exception as e:
            logger.error(f"Validation error for {prop.Name}: {e}")
            # On error, exclude the property (fail-safe: don't return unvalidated)

    logger.info(
        f"Validation: {len(candidates_to_check)} candidates → "
        f"{len(validated)} validated matches"
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
