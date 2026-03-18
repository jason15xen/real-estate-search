"""
Query Parser — Uses Claude to decompose natural language into structured search criteria.

Example: "a house with two bedrooms, a fireplace, and a covered pool within 5 miles of School A"
  → RoomCountCriterion(room_type="Bedroom", exact_count=2)
  → FeatureCriterion(feature="fireplace")
  → FeatureCriterion(feature="covered pool")
  → ProximityCriterion(landmark_name="School A", max_distance_miles=5)
"""

import json
import logging

from config.settings import settings
from src.llm_client import get_async_client_fast
from src.models.search import (
    AreaCriterion,
    FeatureCriterion,
    LocationCriterion,
    ParsedQuery,
    PriceCriterion,
    ProximityCriterion,
    RoomCountCriterion,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a real estate search query parser. Your job is to extract structured \
search criteria from natural language queries about real estate properties.

Extract ALL criteria from the user's query and return them as JSON.

Available criterion types:

1. room_count — The user wants a specific number of rooms.
   Fields: room_type (string), exact_count (int|null), min_count (int|null), max_count (int|null)
   Room types: Bedroom, Bathroom, Kitchen, Living Room, Dining Room, Garage

2. feature — The user wants a specific feature in the property.
   Fields: feature (string), room_context (string|null)
   room_context ties the feature to a specific room type when the user says e.g. \
"a bedroom with a fireplace" — feature="fireplace", room_context="Bedroom".
   If the feature is general (e.g. "a covered pool"), set room_context to null.

3. price — Price range constraint.
   Fields: min_price (int|null), max_price (int|null)

4. area — Square footage constraint.
   Fields: min_sqft (int|null), max_sqft (int|null)

5. location — City/state/country/district constraint.
   Fields: city (string|null), state (string|null), country (string|null), district (string|null)

6. proximity — Distance to a named landmark/place.
   Fields: landmark_name (string), max_distance_miles (float)

Return JSON with this exact structure:
{
  "criteria": [ ... list of criterion objects, each with a "type" field ... ],
  "understood_intent": "Brief summary of what you understood the user is looking for"
}

Important rules:
- If the user says "two bedrooms", that means exact_count=2 for Bedroom.
- If the user says "at least 3 bathrooms", that means min_count=3 for Bathroom.
- Keep feature names concise and lowercase (e.g., "fireplace", "covered pool", "walk-in closet").
- Only extract criteria that are explicitly stated or clearly implied.
- Do NOT invent criteria the user did not mention.
"""


async def parse_query(query: str) -> ParsedQuery:
    client = get_async_client_fast()

    response = await client.messages.create(
        model=settings.anthropic_model_fast,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )

    raw_text = response.content[0].text
    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw_text = "\n".join(lines)

    parsed = json.loads(raw_text)

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
        else:
            logger.warning(f"Unknown criterion type: {criterion_type}")

    return ParsedQuery(
        original_query=query,
        criteria=criteria,
        understood_intent=parsed.get("understood_intent", ""),
    )
