"""
Raw Properties Staging — single source of intake for /process.

Every incoming property record lands here first. A background worker
continuously moves rows from raw_properties → primary tables (properties,
rooms, room_instances, property_schools).

Status lifecycle:
  unprocessed             → new record OR existing record with image URL changes
  image_only_processed    → existing record with no image URL changes (metadata only)
  processed               → fully synced to primary tables; worker has nothing to do

Concurrency notes:
  * upsert_raw_property never downgrades 'unprocessed' to 'image_only_processed'.
    Queued vision work stays queued even if a same-photo update arrives mid-flight.
  * claim_pending_batch returns each row's updated_at; the worker passes it back
    into mark_processed so a concurrent /process write (which bumps updated_at)
    causes mark_processed to no-op — the next iteration picks up the new data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import asyncpg

logger = logging.getLogger(__name__)


def _extract_photo_urls_from_data(data: dict) -> list[str]:
    """Highest-resolution JPEG URL per photo from a `data` payload (a Zillow record).
    Used for the image-change diff in upsert_raw_property().
    """
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


async def upsert_raw_property(
    conn,
    item_id: str,
    new_data: dict,
) -> str:
    """Insert or update a single raw_properties row.

    Decision tree (preserves any pending vision work):
      - row doesn't exist                                → INSERT, status='unprocessed'
      - row exists, photos changed                       → UPDATE, status='unprocessed'
      - row exists, photos unchanged, previously 'unprocessed'
                                                         → UPDATE, status stays 'unprocessed'
      - row exists, photos unchanged, previously 'processed' or 'image_only_processed'
                                                         → UPDATE, status='image_only_processed'

    Returns the final status assigned to the row.
    """
    existing = await conn.fetchrow(
        "SELECT data, status FROM raw_properties WHERE id = $1", item_id
    )
    new_urls = _extract_photo_urls_from_data(new_data)
    new_data_json = json.dumps(new_data)

    if existing is None:
        await conn.execute("""
            INSERT INTO raw_properties (id, data, status)
            VALUES ($1, $2::jsonb, 'unprocessed')
        """, item_id, new_data_json)
        return "unprocessed"

    existing_data = (
        existing["data"] if isinstance(existing["data"], dict) else json.loads(existing["data"])
    )
    existing_urls = _extract_photo_urls_from_data(existing_data)
    photos_changed = new_urls != existing_urls

    # Never drop pending vision work: if previously unprocessed, stay unprocessed.
    if photos_changed or existing["status"] == "unprocessed":
        new_status = "unprocessed"
    else:
        new_status = "image_only_processed"

    await conn.execute("""
        UPDATE raw_properties
        SET data = $2::jsonb, status = $3, updated_at = NOW()
        WHERE id = $1
    """, item_id, new_data_json, new_status)
    return new_status


async def claim_pending_batch(
    conn,
    limit: int = 5,
) -> list[asyncpg.Record]:
    """Claim up to `limit` raw rows that need processing.

    Returns each row's updated_at snapshot so the worker can pass it back into
    mark_processed for optimistic-concurrency guarding. Uses
    SELECT ... FOR UPDATE SKIP LOCKED so multiple concurrent workers don't
    claim the same row.
    """
    rows = await conn.fetch("""
        SELECT id, data, status, updated_at
        FROM raw_properties
        WHERE status IN ('unprocessed', 'image_only_processed')
        ORDER BY updated_at ASC
        LIMIT $1
        FOR UPDATE SKIP LOCKED
    """, limit)
    return rows


async def mark_processed(conn, item_id: str, expected_updated_at: datetime) -> bool:
    """Flip status to 'processed' ONLY if the row hasn't changed since claim.

    Returns True if the row was marked processed, False if a concurrent
    /process write modified the row in the meantime (in which case the new
    data will be re-claimed and re-processed by the next worker iteration).
    """
    result = await conn.execute("""
        UPDATE raw_properties
        SET status = 'processed', updated_at = NOW()
        WHERE id = $1 AND updated_at = $2
    """, item_id, expected_updated_at)
    # asyncpg returns "UPDATE n"
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except ValueError:
        n = 0
    return n > 0


async def get_status_counts(conn) -> dict[str, int]:
    rows = await conn.fetch("SELECT status, COUNT(*) AS n FROM raw_properties GROUP BY status")
    return {r["status"]: r["n"] for r in rows}
