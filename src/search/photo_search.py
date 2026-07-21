"""Photo filtering within a given set of properties (gallery endpoint):
a FIXED query vocabulary maps directly to room types — no LLM involved, so requests
are instant, free, and deterministic.

Supported queries (normalized case/whitespace-insensitively):
    show pools             -> Pool
    show kitchens          -> Kitchen
    show primary-room      -> Bedroom
    show front-exteriors   -> Exterior
    show living-rooms      -> Living Room
    show primary-bathrooms -> Bathroom
    show backyards         -> Garage, Unknown
"""

from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)

QUERY_ROOM_MAP: dict[str, list[str]] = {
    "show pools": ["Pool"],
    "show kitchens": ["Kitchen"],
    "show primary-room": ["Bedroom"],
    "show front-exteriors": ["Exterior"],
    "show living-rooms": ["Living Room"],
    "show primary-bathrooms": ["Bathroom"],
    "show backyards": ["Garage", "Unknown"],
}


class UnsupportedPhotoQueryError(Exception):
    """The query is not one of the fixed supported filters."""

    def __init__(self, query: str):
        supported = ", ".join(sorted(QUERY_ROOM_MAP))
        super().__init__(f"Unsupported query {query!r}. Supported queries: {supported}")


async def search_photos(
    pool: asyncpg.Pool, query: str, property_guids: list[str]
) -> list[dict]:
    """Return [{"id": guid, "imageUrl": [urls]}] for properties (among the given ones)
    that have photos of the query's room types; properties without matches are omitted,
    input order is preserved."""
    key = " ".join(query.strip().lower().split())
    room_types = QUERY_ROOM_MAP.get(key)
    if room_types is None:
        raise UnsupportedPhotoQueryError(query)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT p.guid, ri.photo_url
            FROM room_instances ri
            JOIN properties p ON p.id = ri.property_id
            WHERE p.guid = ANY($1)
              AND ri.photo_url IS NOT NULL
              AND ri.room_type = ANY($2)
            ORDER BY p.guid, ri.room_type, ri.instance_index
            """,
            property_guids,
            room_types,
        )

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
