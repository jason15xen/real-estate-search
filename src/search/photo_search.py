"""Photo-level search within a given set of properties (gallery endpoint):
"show me kitchens" + [guids] → matching photo URLs per property.

Reuses the LLM query parser and the embedding feature resolver, but matches PER PHOTO
(a photo qualifies by its own room type / color / features), unlike the main /search
which matches per property (e.g. dominant color). Negated criteria are ignored — a
gallery wants "photos OF x", not "properties without x"."""

from __future__ import annotations

import logging

import asyncpg

from config.settings import settings
from src.data.feature_registry import registry
from src.models.search import ColorRoomCriterion, FeatureCriterion, RoomCountCriterion
from src.search.query_parser import parse_query

logger = logging.getLogger(__name__)


class NoPhotoCriteriaError(Exception):
    """The query contained nothing that can select photos (no room/color/feature)."""


async def search_photos(
    pool: asyncpg.Pool, query: str, property_guids: list[str]
) -> list[dict]:
    """Return [{"id": guid, "imageUrl": [urls]}] for properties (among the given ones)
    that have photos matching the query; properties without matches are omitted."""
    parsed = await parse_query(query)

    # Photo selectors: (room_type, color|None) pairs, OR-ed — the pairing must stay
    # together per criterion, or "golden kitchens and blue bathrooms" would wrongly
    # also match blue kitchens / gold bathrooms (cross-product).
    pairs: list[tuple[str, str | None]] = []
    feature_phrases: list[str] = []
    for c in parsed.criteria:
        if isinstance(c, ColorRoomCriterion) and not c.negated:
            pairs.append((c.room_type, c.color))
        elif isinstance(c, RoomCountCriterion):
            if c.exact_count == 0 or c.max_count == 0:
                continue  # "without a kitchen" — not a photo selector
            pairs.append((c.room_type, None))
        elif isinstance(c, FeatureCriterion) and not c.negated:
            feature_phrases.append(c.feature)
            if c.room_context:
                pairs.append((c.room_context, None))

    pairs = list(dict.fromkeys(pairs))
    if not pairs and not feature_phrases:
        raise NoPhotoCriteriaError(
            "Query contains no room type, color, or feature to match photos on"
        )

    # Each feature phrase becomes an OR-list of canonical DB feature strings
    # (embedding retrieve-then-rerank when enabled, else registry word-matching),
    # AND-ed across phrases — mirroring the main search's feature phase, per photo.
    feature_lists: list[list[str]] = []
    if feature_phrases:
        resolved: dict[str, list[str]] = {}
        if settings.search_use_embedding_retrieval:
            from src.search.feature_resolver import resolve_feature_phrases
            try:
                resolved = await resolve_feature_phrases(pool, feature_phrases)
            except Exception as e:  # noqa: BLE001 — fall back to registry matching
                logger.warning(f"Photo search: feature resolution failed ({e}); using registry")
        for phrase in feature_phrases:
            alts = resolved.get(phrase) or registry.get_feature_alternatives(phrase)
            feature_lists.append(list({phrase, *alts}))

    conditions = ["p.guid = ANY($1)", "ri.photo_url IS NOT NULL"]
    params: list = [property_guids]
    idx = 2
    if pairs:
        selector_sql: list[str] = []
        for room, color in pairs:
            if color is None:
                selector_sql.append(f"ri.room_type = ${idx}")
                params.append(room)
                idx += 1
            else:
                selector_sql.append(f"(ri.room_type = ${idx} AND ri.color = ${idx + 1})")
                params.append(room)
                params.append(color)
                idx += 2
        conditions.append("(" + " OR ".join(selector_sql) + ")")
    for flist in feature_lists:
        conditions.append(f"ri.features && ${idx}::text[]")
        params.append(flist)
        idx += 1

    sql = f"""
        SELECT p.guid, ri.photo_url
        FROM room_instances ri
        JOIN properties p ON p.id = ri.property_id
        WHERE {" AND ".join(conditions)}
        ORDER BY p.guid, ri.room_type, ri.instance_index
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    by_guid: dict[str, list[str]] = {}
    for r in rows:
        urls = by_guid.setdefault(r["guid"], [])
        if r["photo_url"] not in urls:  # dedupe defensively
            urls.append(r["photo_url"])
    # Preserve the caller's property order; omit properties with no matches.
    return [
        {"id": g, "imageUrl": by_guid[g]}
        for g in dict.fromkeys(property_guids)  # order-preserving dedupe of input
        if g in by_guid
    ]
