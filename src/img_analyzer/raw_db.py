"""
Raw Properties Staging — intake for /process. A background worker moves rows
from raw_properties → primary tables (properties, rooms, room_instances,
property_schools).

Status lifecycle:
  unprocessed                  → new, or all photos changed / no primary row; re-analyze all.
  image_only_processed         → photo set identical; refresh scalars+schools only, no Vision.
  partial_image_only_processed → photos partially changed; analyze added, drop removed.
  processed                    → fully synced; nothing to do.

No stored diff: truth is raw_properties.data (desired) vs room_instances.photo_url
(current). Diff is set arithmetic, computed transiently in upsert and recomputed by
the worker at processing time, keeping the worker idempotent under concurrent writes.

Concurrency invariants:
  * upsert never downgrades away from 'unprocessed' (preserves queued vision work).
  * mark_processed no-ops if updated_at changed since claim, so concurrent writes are reprocessed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import asyncpg

logger = logging.getLogger(__name__)


def extract_photo_urls_from_data(data: dict) -> list[str]:
    """Highest-res JPEG URL per photo; same extractor as room_instances.photo_url so diffs round-trip."""
    urls: list[str] = []
    for photo in (data.get("originalPhotos") or []):
        jpegs = ((photo.get("mixedSources") or {}).get("jpeg") or [])
        if not jpegs:
            continue
        best = max(jpegs, key=lambda j: j.get("width", 0))
        url = best.get("url", "")
        if url:
            urls.append(url)
    return urls


async def primary_photo_urls(conn, property_guid: str) -> set[str]:
    """Photo URLs already in primary (room_instances.photo_url). Empty if absent or
    legacy NULL photo_url rows (treated as unknown → full re-analyze)."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT ri.photo_url
        FROM room_instances ri
        JOIN properties p ON p.id = ri.property_id
        WHERE p.guid = $1 AND ri.photo_url IS NOT NULL
        """,
        property_guid,
    )
    return {r["photo_url"] for r in rows}


async def upsert_raw_property(
    conn,
    item_id: str,
    new_data: dict,
) -> tuple[str, dict | None, bool]:
    """Insert or update a raw_properties row.

    Returns (final_status, diff_info_or_None, existed_before). diff_info is for the
    /process response only (not stored; worker recomputes against live primary state).

    Status: no row → unprocessed; no primary URLs → unprocessed; photos identical →
    image_only_processed; partial overlap → partial_image_only_processed; no overlap →
    unprocessed. Never downgrades a prior 'unprocessed' (preserves queued vision work).
    """
    existing = await conn.fetchrow(
        "SELECT status FROM raw_properties WHERE id = $1", item_id
    )
    new_urls = extract_photo_urls_from_data(new_data)
    new_data_json = json.dumps(new_data)
    new_url_set = set(new_urls)

    if existing is None:
        await conn.execute(
            """
            INSERT INTO raw_properties (id, data, status)
            VALUES ($1, $2::jsonb, 'unprocessed')
            """,
            item_id,
            new_data_json,
        )
        return "unprocessed", None, False

    primary_urls = await primary_photo_urls(conn, item_id)

    added = sorted(new_url_set - primary_urls)
    removed = sorted(primary_urls - new_url_set)
    unchanged = new_url_set & primary_urls

    if not primary_urls:
        new_status = "unprocessed"
        diff_info: dict | None = None
    elif not added and not removed:
        new_status = "image_only_processed"
        diff_info = None
    elif not unchanged:
        new_status = "unprocessed"
        diff_info = None
    else:
        new_status = "partial_image_only_processed"
        diff_info = {"added": added, "removed": removed}

    # Never silently drop queued vision work.
    if existing["status"] == "unprocessed":
        new_status = "unprocessed"
        diff_info = None

    await conn.execute(
        """
        UPDATE raw_properties
        SET data = $2::jsonb,
            status = $3,
            updated_at = NOW()
        WHERE id = $1
        """,
        item_id,
        new_data_json,
        new_status,
    )
    return new_status, diff_info, True


async def claim_pending_batch(
    conn,
    limit: int = 5,
) -> list[asyncpg.Record]:
    """Claim up to `limit` pending rows via FOR UPDATE SKIP LOCKED. Returns updated_at
    for optimistic-concurrency guarding in mark_processed."""
    rows = await conn.fetch(
        """
        SELECT id, data, status, updated_at
        FROM raw_properties
        WHERE status IN (
            'unprocessed',
            'image_only_processed',
            'partial_image_only_processed'
        )
        ORDER BY updated_at ASC
        LIMIT $1
        FOR UPDATE SKIP LOCKED
        """,
        limit,
    )
    return rows


async def mark_processed(conn, item_id: str, expected_updated_at: datetime) -> bool:
    """Set status='processed' only if updated_at is unchanged since claim. Returns False
    if a concurrent write moved it; the row stays pending for the next (idempotent) run."""
    result = await conn.execute(
        """
        UPDATE raw_properties
        SET status = 'processed', updated_at = NOW()
        WHERE id = $1 AND updated_at = $2
        """,
        item_id,
        expected_updated_at,
    )
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except ValueError:
        n = 0
    return n > 0


async def get_status_counts(conn) -> dict[str, int]:
    rows = await conn.fetch("SELECT status, COUNT(*) AS n FROM raw_properties GROUP BY status")
    return {r["status"]: r["n"] for r in rows}
