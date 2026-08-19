"""Detailed property photos: ALL images of the given properties, grouped by the
vision-derived room type — the same {roomType, urls} structure /search returns in
briefproperties.propertyphotos, including the synthetic "Main" highlight group
(first image of each room type, Exterior leading so Main[0] is the house front).

Difference from the search listing: urls here are the FULL-RESOLUTION jpegs —
this is the detail/lightbox endpoint; the search list serves smallest-res
thumbnails. No LLM involved — instant and free.
"""

from __future__ import annotations

import logging

import asyncpg

from src.search.orchestrator import _photo_groups

logger = logging.getLogger(__name__)


async def detailed_photos(
    pool: asyncpg.Pool, property_guids: list[str]
) -> list[dict]:
    """[{"id": guid, "propertyphotos": [{"roomType", "urls"}, ...]}, ...] for the
    requested properties, input order preserved (input duplicates collapsed);
    unknown guids are omitted."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT p.id AS pid, p.guid, r.data->>'originalPhotos' AS photos_json
            FROM properties p
            LEFT JOIN raw_properties r ON r.id = p.guid
            WHERE p.guid = ANY($1)
            """,
            property_guids,
        )
        room_rows = await conn.fetch(
            """
            SELECT property_id, photo_url,
                   CASE WHEN room_type = 'Pool' AND EXISTS (SELECT 1 FROM unnest(features) cf WHERE cf ILIKE '%community%pool%')
                        THEN 'Community Pool' ELSE room_type END AS room_type,
                   (room_type = 'Pool'
                    AND features && ARRAY['pool','covered pool','uncovered pool']) AS pool_water
            FROM room_instances
            WHERE property_id = ANY($1) AND photo_url IS NOT NULL
            """,
            [r["pid"] for r in rows],
        )

    room_map: dict[int, dict[str, str]] = {}
    water_map: dict[int, set[str]] = {}
    for rr in room_rows:
        room_map.setdefault(rr["property_id"], {})[rr["photo_url"]] = rr["room_type"]
        if rr["pool_water"]:
            water_map.setdefault(rr["property_id"], set()).add(rr["photo_url"])

    by_guid = {r["guid"]: r for r in rows}
    out = []
    for guid in dict.fromkeys(property_guids):  # order-preserving input dedupe
        r = by_guid.get(guid)
        if r is None:
            continue  # unknown guid — omitted
        out.append({
            "id": guid,
            "propertyphotos": _photo_groups(
                r["photos_json"], room_map.get(r["pid"], {}), full_size=True,
                water_urls=water_map.get(r["pid"]),
            ),
        })
    return out
