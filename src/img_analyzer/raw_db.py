"""
Raw Properties Staging — single source of intake for /process.

Every incoming property record lands here first. A background worker
continuously moves rows from raw_properties → primary tables (properties,
rooms, room_instances, property_schools).

Status lifecycle:
  unprocessed                  → new record, OR existing one where ALL photos
                                  changed / there's no primary row yet — worker
                                  re-analyzes every photo.
  image_only_processed         → existing record, photo set IDENTICAL to what's
                                  already in primary — worker refreshes scalars
                                  and schools only, no Vision API.
  partial_image_only_processed → existing record, photo set CHANGED but only
                                  partially — worker recomputes the diff at
                                  processing time against current primary
                                  state, analyzes only the newly-added photos
                                  and deletes room_instances for removed ones.
  processed                    → fully synced to primary tables; worker has
                                  nothing to do.

Design note — no stored diff:
  We deliberately do NOT persist the photo diff. The truth lives in two
  places already: raw_properties.data (desired photo set) and
  room_instances.photo_url (currently processed photo set). The diff is just
  set arithmetic over those — computed transiently in upsert (to pick the
  right status + report counts in the /process response) and recomputed
  freshly by the worker at processing time. That makes the worker idempotent
  under concurrent /process writes: regardless of how many partial updates
  pile up while one is mid-flight, each worker run only does work that's
  actually still needed.

Concurrency invariants:
  * upsert never DOWNGRADES away from 'unprocessed' — if the worker hasn't
    yet seen a queued full-process, the next request preserves it.
  * claim_pending_batch returns each row's updated_at; the worker passes it
    back into mark_processed so a concurrent /process write (which bumps
    updated_at) causes mark_processed to no-op — the next iteration picks
    up the new data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import asyncpg

logger = logging.getLogger(__name__)


def extract_photo_urls_from_data(data: dict) -> list[str]:
    """Highest-resolution JPEG URL per photo from a `data` payload (a Zillow
    record). This is the same canonical-URL extractor used to populate
    room_instances.photo_url, so diffs round-trip without normalization
    issues.
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


# Back-compat alias (older code referenced the private name).
_extract_photo_urls_from_data = extract_photo_urls_from_data


async def primary_photo_urls(conn, property_guid: str) -> set[str]:
    """The set of photo URLs already represented in the primary tables for
    this property — read from room_instances.photo_url.

    Returns an empty set if the property doesn't exist in primary yet, OR if
    it exists but its rooms were created before photo_url tracking (legacy
    rows have NULL photo_url, so we treat their photo set as unknown and
    fall back to full re-analyze).
    """
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
) -> tuple[str, dict | None]:
    """Insert or update a single raw_properties row.

    Returns (final_status, diff_info_or_None). The diff_info is only used by
    the caller (the /process router) to report per-id counts in the response
    — it is NOT stored. The worker recomputes the diff itself at processing
    time, against live primary state.

    Decision tree:
      - row doesn't exist                                → INSERT, status='unprocessed'
      - primary has no processed photo URLs yet          → UPDATE, status='unprocessed'
      - photos identical to primary                      → UPDATE, status='image_only_processed'
      - photos partially different                       → UPDATE, status='partial_image_only_processed'
      - photos completely different (no overlap)         → UPDATE, status='unprocessed'

    Override: if the previous raw status was 'unprocessed' (worker hasn't
    yet consumed the queued full analyze), we DON'T downgrade — pending
    vision work is preserved.
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
        return "unprocessed", None

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
    return new_status, diff_info


async def claim_pending_batch(
    conn,
    limit: int = 5,
) -> list[asyncpg.Record]:
    """Claim up to `limit` raw rows that need processing.

    Uses SELECT ... FOR UPDATE SKIP LOCKED so concurrent workers don't claim
    the same row. Returns each row's `updated_at` so the worker can pass it
    back into mark_processed for optimistic-concurrency guarding.
    """
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
    """Flip status to 'processed' ONLY if the row hasn't changed since claim.

    Returns True on success, False if a concurrent /process write moved
    updated_at — in which case the row stays pending and the next worker
    iteration re-processes it (idempotently, since the worker recomputes
    the diff itself).
    """
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
