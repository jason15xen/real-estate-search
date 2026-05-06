"""
Query Parser — Uses Azure OpenAI to decompose natural language into structured search criteria.

The parser receives the full list of known features and room types from the data,
so it can map user input to exact feature names. This eliminates the need for
vector search and LLM validation.

Example: "a house with two bedrooms, wood flooring, and a hearth"
  → RoomCountCriterion(room_type="Bedroom", exact_count=2)
  → FeatureCriterion(feature="hardwood floors")   ← mapped from "wood flooring"
  → FeatureCriterion(feature="fireplace")          ← mapped from "hearth"
"""

import json
import logging

from config.settings import settings
from src.data.feature_registry import registry
from src.llm_client import get_async_client
from src.models.search import (
    AreaCriterion,
    FeatureCriterion,
    LocationCriterion,
    ParsedQuery,
    PriceCriterion,
    PropertyCriterion,
    ProximityCriterion,
    RoomCountCriterion,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are a real estate search query parser. Your job is to extract structured \
search criteria from natural language queries about real estate properties.

You have access to the EXACT feature names and room types that exist in our database. \
You MUST map the user's words to these exact names.

KNOWN ROOM TYPES:
{room_types}

KNOWN FEATURES (these are the ONLY valid feature names):
{features}

Extract ALL criteria from the user's query and return them as JSON.

Available criterion types:

1. room_count — The user wants a specific number of rooms.
   Fields: room_type (string), exact_count (int|null), min_count (int|null), max_count (int|null)
   Use ONLY the room types listed above.

2. feature — The user wants a specific feature included or excluded.
   Fields: feature (string), room_context (string|null), negated (bool)
   CRITICAL: Map the user's words to the closest matching known feature name. \
But if the user's word is a GENERIC term that could match MANY features \
(e.g., "cabinet", "tile", "wood", "pool"), keep the generic term as-is \
so it matches broadly. Only map to a specific feature when the user is clearly specific.
   Examples of mapping:
     "wood flooring" → "hardwood floors" (specific → specific)
     "hearth" → "fireplace" (specific → specific)
     "marble counters" → "marble countertops" (specific → specific)
     "cabinet" → "cabinet" (generic → keep generic, matches "white cabinets", "shaker cabinets", etc.)
     "tile" → "tile" (generic → keep generic, matches "tile flooring", "tile backsplash", etc.)
     "pool" → "pool" (generic → keep generic)
   If the user mentions a feature that has NO close match in the known features, \
still include it using the user's original wording. NEVER drop a feature from the query.
   room_context ties the feature to a specific room type. Set it ONLY when \
the user EXPLICITLY names a room type as a CONTAINER for the feature \
(pattern: "<ROOM TYPE> with <FEATURE>" or "<FEATURE> in the <ROOM TYPE>").
   DO NOT set room_context just because a feature shares its name with a room type \
(e.g. "pool", "kitchen", "garage" used as a FEATURE). These refer to the feature itself, \
not a containing room.
   Correct usage:
     "kitchen with granite countertops"   → feature=granite countertops, room_context="Kitchen"
     "bedrooms with accent walls"         → feature=accent walls, room_context="Bedroom"
     "hardwood floors in the living room" → feature=hardwood floors, room_context="Living Room"
   Incorrect usage (DO NOT DO THIS):
     "with pool"                → room_context=null   (NOT "Pool")
     "with covered pool"        → room_context=null   (NOT "Pool")
     "with uncovered pool"      → room_context=null   (NOT "Pool")
     "with garage"              → room_context=null   (NOT "Garage")
     "3 bedrooms with a fireplace" → room_context=null (bedroom count is separate; fireplace is not IN the bedroom here)
   negated=true means the property must NOT have this feature.

   NEGATION SEMANTICS — READ CAREFULLY:

   There are TWO KINDS of negation. You MUST distinguish them:

   (A) ATOMIC NEGATION — the user doesn't want the thing at all.
       Phrasing cues: "no X", "without X", "not X", "excluding X" \
when X stands alone with no modifier.
       Emit ONE negated criterion for X only.
       Examples:
         "no pool"           → [negated(pool)]
         "without garage"    → [negated(garage)]
         "not waterfront"    → [negated(waterfront)]

   (B) MODIFIER NEGATION — the user DOES want the thing, but NOT with a specific attribute.
       Phrasing cues: a compound noun where a negative prefix/word modifies \
an attribute of a thing the user wants (e.g., "uncovered pool", "unfenced yard", \
"open-air pool", "gas-free fireplace", "pool without cover", "fireplace without gas").
       Emit TWO criteria:
         1. POSITIVE criterion for the BASE thing (it must exist)
         2. NEGATED criterion for the unwanted ATTRIBUTE
       Examples:
         "with uncovered pool"       → [positive(pool), negated(covered pool)]
         "pool without screen"       → [positive(pool), negated(screened pool)]
         "open-air pool"             → [positive(pool), negated(covered pool)]
         "fireplace without gas"     → [positive(fireplace), negated(gas fireplace)]
         "unfenced backyard"         → [positive(backyard), negated(fenced backyard)]
         "non-granite countertops"   → [positive(countertops), negated(granite countertops)]

   HOW TO DECIDE (A) vs (B):
     - If the negated word stands alone with no described object → (A) atomic
     - If the negation attaches to a MODIFIER of a thing the user wants → (B) modifier
     - Rule of thumb: if removing the negation still leaves something the user wants \
(e.g., "uncovered pool" → still wants a pool), it's MODIFIER NEGATION.
     - If removing the negation leaves nothing wanted (e.g., "no pool" → wants no pool), \
it's ATOMIC NEGATION.

   NEVER expand negations to related/similar features the user did not mention. \
For example, "without granite" must NOT also negate "quartz" — only the exact thing \
the user said.

3. price — Price range constraint.
   Fields: min_price (int|null), max_price (int|null)

4. area — Square footage constraint.
   Fields: min_sqft (int|null), max_sqft (int|null)

5. location — City/state/country/district constraint.
   Fields: city (string|null), state (string|null), country (string|null), district (string|null)

6. proximity — Distance to a SPECIFIC named landmark/place.
   Fields: landmark_name (string), max_distance_miles (float)
   For "near good schools", use landmark_name="good schools" and max_distance_miles=5.
   ONLY use proximity for specific named places (e.g. "Oak Park Elementary School", "Central Park").
   Do NOT use proximity for generic features like "beach", "lake", "park", "downtown".
   Instead, use "feature" type for these: "near beach" → feature="beach", \
"near lake" → feature="lake", "near park" → feature="park".

7. property — Property attribute constraints.
   Fields: home_type (string|null), min_rent (int|null), max_rent (int|null), \
min_year_built (int|null), max_year_built (int|null), \
min_lot_sqft (int|null), max_lot_sqft (int|null), \
min_stories (int|null), max_stories (int|null)
   VALID home_type values: SINGLE_FAMILY, CONDO, TOWNHOUSE, MANUFACTURED, MULTI_FAMILY
   Only set home_type when the user's term CLEARLY maps to one of these values.
   If ambiguous (e.g. "apartment", "home", "house", "property"), do NOT set home_type.
   Mapping:
     "condo" → CONDO
     "townhouse" → TOWNHOUSE
     "single family" / "single-family home" / "family home" / "family house" / "family residence" / "starter home" → SINGLE_FAMILY
     "manufactured home" / "mobile home" → MANUFACTURED
     "duplex" / "multi family" / "multi-family" / "two-family" → MULTI_FAMILY
   IMPORTANT: When the user describes a HOME TYPE phrase (e.g. "family home", "starter home", \
"single-family residence"), emit it ONLY as a `property` criterion with `home_type` set. \
Do NOT also emit a `feature` criterion for the same phrase. \
"family home" is a home type, NOT a feature like "family-friendly community".
   "under $2k/mo" or "rent under 2000" → max_rent=2000
   "built after 2000" → min_year_built=2000
   "single story" → max_stories=1
   IMPORTANT: There is NO has_pool or has_waterfront field on the `property` criterion.
   "pool", "swimming pool", "waterfront", "on the water" are FEATURES. \
Use the `feature` criterion type for them — never emit them as property attributes.

Return JSON with this exact structure:
{{
  "criteria": [ ... list of criterion objects, each with a "type" field ... ],
  "reconstructed_queries": [],
  "understood_intent": "Brief summary of what you understood the user is looking for"
}}

IMPORTANT: `reconstructed_queries` is DEPRECATED. Always return an empty array `[]` \
for this field. Do NOT expand synonyms or related features — the server computes \
feature alternatives deterministically from the database. Returning anything other \
than `[]` is wasted output that will be discarded.

  User: "3 bedroom home with granite counters"
  KNOWN FEATURES contain: "granite countertops"
  reconstructed_queries: ["granite countertops"]

  User: "kitchen with tile floor"
  KNOWN FEATURES contain: "tile flooring", "porcelain tile flooring", "ceramic tile flooring"
  reconstructed_queries: ["tile flooring", "porcelain tile flooring", "ceramic tile flooring"]

  User: "house with wood floors and a tub"
  KNOWN FEATURES contain: "hardwood flooring", "wood flooring", "soaking tub", "freestanding tub"
  reconstructed_queries: ["hardwood flooring", "wood flooring", "soaking tub", "freestanding tub"]

Important rules:
- If the user says "two bedrooms", that means exact_count=2 for Bedroom.
- If the user says "at least 3 bathrooms", that means min_count=3 for Bathroom.
- CRITICAL: Feature values MUST be from the KNOWN FEATURES list above. \
Map synonyms, abbreviations, and alternate phrasings to the exact known feature name.
- Only extract criteria that are explicitly stated or clearly implied.
- Do NOT invent criteria the user did not mention.
- room_context rules (see FEATURE section above for the full rule):
  Set room_context ONLY when the user names a room type as a CONTAINER via the pattern \
"<ROOM TYPE> with <FEATURE>" or "<FEATURE> in the <ROOM TYPE>".
  NEVER set room_context because a feature name coincides with a room type. \
For "pool", "garage", "kitchen", "bedroom" used as features, room_context=null.
- When the user says "without", "no", "exclude", or "not", set negated=true.
  "2 bedrooms without stone tile" → feature="stone tile", negated=true
  "no pool" → the closest known feature, negated=true
"""


def _build_system_prompt() -> str:
    features = registry.get_features_list()
    room_types = registry.get_room_types_list()
    return SYSTEM_PROMPT_TEMPLATE.format(
        room_types=", ".join(room_types),
        features=", ".join(features),
    )


async def _call_llm(client, system_prompt: str, query: str) -> str | None:
    """Call Azure OpenAI and return raw text. Returns None on failure."""
    response = await client.chat.completions.create(
        model=settings.azure_openai_deployment,
        max_completion_tokens=16384,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    raw_text = response.choices[0].message.content
    if not raw_text or not raw_text.strip():
        return None
    return raw_text.strip()


async def parse_query(query: str, max_retries: int = 2) -> ParsedQuery:
    client = get_async_client()
    system_prompt = _build_system_prompt()

    raw_text = None
    parsed = None

    for attempt in range(max_retries):
        try:
            raw_text = await _call_llm(client, system_prompt, query)
            if not raw_text:
                logger.warning(f"LLM returned empty response (attempt {attempt + 1}/{max_retries})")
                continue

            logger.debug(f"Raw LLM response: {raw_text[:500]}")

            # Strip markdown code fences if present
            clean = raw_text
            if clean.startswith("```"):
                lines = clean.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                clean = "\n".join(lines).strip()

            parsed = json.loads(clean)
            break  # Success
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(
                f"Failed to parse LLM response (attempt {attempt + 1}/{max_retries}): {e}\n"
                f"Raw text: {raw_text[:500] if raw_text else 'None'}"
            )
            continue
        except Exception as e:
            logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            continue

    if not parsed:
        logger.error(f"All {max_retries} attempts failed for query: '{query}'")
        return ParsedQuery(
            original_query=query,
            criteria=[],
            understood_intent="Failed to parse query after retries",
        )

    try:
        criteria = []
        for c in parsed["criteria"]:
            criterion_type = c["type"]
            if criterion_type == "room_count":
                criteria.append(RoomCountCriterion(
                    room_type=c["room_type"],
                    exact_count=c.get("exact_count"),
                    min_count=c.get("min_count"),
                    max_count=c.get("max_count"),
                ))
            elif criterion_type == "feature":
                criteria.append(FeatureCriterion(
                    feature=c["feature"],
                    room_context=c.get("room_context"),
                    negated=c.get("negated", False),
                ))
            elif criterion_type == "price":
                criteria.append(PriceCriterion(
                    min_price=c.get("min_price"),
                    max_price=c.get("max_price"),
                ))
            elif criterion_type == "area":
                criteria.append(AreaCriterion(
                    min_sqft=c.get("min_sqft"),
                    max_sqft=c.get("max_sqft"),
                ))
            elif criterion_type == "location":
                criteria.append(LocationCriterion(
                    city=c.get("city"),
                    state=c.get("state"),
                    country=c.get("country"),
                    district=c.get("district"),
                ))
            elif criterion_type == "proximity":
                criteria.append(ProximityCriterion(
                    landmark_name=c["landmark_name"],
                    max_distance_miles=c["max_distance_miles"],
                ))
            elif criterion_type == "property":
                criteria.append(PropertyCriterion(
                    home_type=c.get("home_type"),
                    min_rent=c.get("min_rent"),
                    max_rent=c.get("max_rent"),
                    min_year_built=c.get("min_year_built"),
                    max_year_built=c.get("max_year_built"),
                    min_lot_sqft=c.get("min_lot_sqft"),
                    max_lot_sqft=c.get("max_lot_sqft"),
                    min_stories=c.get("min_stories"),
                    max_stories=c.get("max_stories"),
                ))
            else:
                logger.warning(f"Unknown criterion type: {criterion_type}")
    except (KeyError, TypeError) as e:
        logger.error(f"Failed to extract criteria: {e}")
        return ParsedQuery(
            original_query=query,
            criteria=[],
            understood_intent="Failed to parse query criteria",
        )

    return ParsedQuery(
        original_query=query,
        criteria=criteria,
        reconstructed_queries=parsed.get("reconstructed_queries", []),
        understood_intent=parsed.get("understood_intent", ""),
    )
