"""
Background Worker — continuously drains raw_properties → primary tables.

Picks up rows with status 'unprocessed' or 'image_only_processed', does the
appropriate amount of work, and marks them 'processed'.

Lifecycle:
  - Started by FastAPI lifespan on app startup.
  - Loops forever; processes WORKER_BATCH_SIZE rows in parallel each iteration.
  - On error for a single row, leaves its status untouched so the next pass
    retries (per Q3=a: failures naturally retry).

Concurrency model (per Q5=b):
  asyncio.gather() over WORKER_BATCH_SIZE rows. Each row takes its own DB
  connection from the pool. Vision API calls within a single row remain
  internally concurrent (5 photos at a time via analyze_photos).
"""

from __future__ import annotations

import asyncio
import json
import logging

import asyncpg

from src.data.database import get_pool
from src.img_analyzer.analyzer import analyze_photos, inject_features
from src.img_analyzer.db_ingest import (
    _build_rooms_from_photos,
    _extract_property_fields,
    _extract_schools,
    _get_room_counts,
    _insert_children,
    update_property_metadata,
    update_property_with_children,
)
from src.img_analyzer.models import PropertyItem
from src.img_analyzer.raw_db import claim_pending_batch, mark_processed

logger = logging.getLogger(__name__)

WORKER_BATCH_SIZE = 3            # number of rows processed in parallel per iteration
WORKER_IDLE_SLEEP_SECONDS = 5    # sleep when there's no work to do
WORKER_ERROR_SLEEP_SECONDS = 10  # sleep after an iteration that errored


async def _process_unprocessed(
    pool: asyncpg.Pool, item_id: str, data: dict, expected_updated_at,
) -> bool:
    """Full sync: Vision API → primary tables (INSERT or full re-INSERT of children).

    Returns True if mark_processed succeeded, False if a concurrent /process
    write modified the row in raw_properties — in which case status stays
    pending so the next worker iteration re-processes with the fresh data.
    """
    item = {"Id": item_id, "ZillowPropertyRecord": data}

    # Validate; if structure is broken let the caller catch & log
    p = PropertyItem(**item)
    photos = p.ZillowPropertyRecord.originalPhotos

    if photos:
        results = await analyze_photos(property_id=p.Id, photos=photos)
        inject_features([item], {p.Id: results})

    async with pool.acquire() as conn:
        existing_id = await conn.fetchval(
            "SELECT id FROM properties WHERE guid = $1", item_id
        )
        if existing_id is None:
            await _insert_primary_full(conn, item)
        else:
            await update_property_with_children(conn, existing_id, item)
        return await mark_processed(conn, item_id, expected_updated_at)


async def _process_image_only(
    pool: asyncpg.Pool, item_id: str, data: dict, expected_updated_at,
) -> bool:
    """Metadata-only sync: update properties row + schools, leave room_instances alone.

    If the primary row is missing, falls back to the full unprocessed path so
    vision analysis runs (otherwise we'd insert a property with zero rooms).
    Returns True if mark_processed succeeded.
    """
    async with pool.acquire() as conn:
        existing_id = await conn.fetchval(
            "SELECT id FROM properties WHERE guid = $1", item_id
        )
    if existing_id is None:
        return await _process_unprocessed(pool, item_id, data, expected_updated_at)

    item = {"Id": item_id, "ZillowPropertyRecord": data}
    async with pool.acquire() as conn:
        await update_property_metadata(conn, existing_id, item)
        await conn.execute("DELETE FROM property_schools WHERE property_id = $1", existing_id)
        for s in _extract_schools(item):
            await conn.execute("""
                INSERT INTO property_schools (
                    property_id, school_name, rating, grades, distance_miles, link
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, existing_id, s["name"], s["rating"], s["grades"], s["distance"], s["link"])
        return await mark_processed(conn, item_id, expected_updated_at)


async def _insert_primary_full(conn, item: dict) -> None:
    """INSERT a brand-new property + all its child rows. Used by both
    'unprocessed' new arrivals and the rare 'image_only_processed' case where
    the primary row is somehow missing.
    """
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    room_counts = _get_room_counts(record, rooms_from_photos)

    prop_id = await conn.fetchval("""
        INSERT INTO properties (
            guid, name, street, district, city, state, postal_code, country,
            geom, area_sqft, price_usd,
            bedroom_count, bathroom_count, kitchen_count,
            living_room_count, dining_room_count, garage_count,
            home_type, rent_estimate, year_built,
            lot_size_sqft, stories,
            has_pool, has_waterfront, description, financing
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8,
            ST_MakePoint($9, $10)::geography,
            $11, $12, $13, $14, $15, $16, $17, $18,
            $19, $20, $21, $22, $23, $24, $25, $26, $27
        ) RETURNING id
    """,
        fields["guid"], fields["name"], fields["street"], fields["district"],
        fields["city"], fields["state"], fields["postal_code"], fields["country"],
        fields["longitude"], fields["latitude"],
        fields["area_sqft"], fields["price_usd"],
        room_counts.get("Bedroom", 0), room_counts.get("Bathroom", 0),
        room_counts.get("Kitchen", 0), room_counts.get("Living Room", 0),
        room_counts.get("Dining Room", 0), room_counts.get("Garage", 0),
        fields["home_type"], fields["rent_estimate"], fields["year_built"],
        fields["lot_size_sqft"], fields["stories"],
        fields["has_pool"], fields["has_waterfront"], fields["description"],
        fields["financing"],
    )
    await _insert_children(conn, prop_id, rooms_from_photos, _extract_schools(item))


async def _process_one_row(pool: asyncpg.Pool, row: asyncpg.Record) -> bool:
    """Dispatch a single raw row to the right processor.

    Returns True if the row was successfully marked processed, False if it
    was raced (concurrent /process update) or if processing raised. Either
    way the row remains pending in raw_properties for the next iteration.
    """
    item_id = row["id"]
    raw_data = row["data"]
    if not isinstance(raw_data, dict):
        raw_data = json.loads(raw_data)
    status = row["status"]
    expected_updated_at = row["updated_at"]
    try:
        if status == "unprocessed":
            marked = await _process_unprocessed(pool, item_id, raw_data, expected_updated_at)
        elif status == "image_only_processed":
            marked = await _process_image_only(pool, item_id, raw_data, expected_updated_at)
        else:
            return False
        if marked:
            logger.info(f"Worker processed {item_id} (was {status})")
        else:
            logger.info(
                f"Worker re-queue {item_id}: concurrent /process write since claim"
            )
        return marked
    except Exception as e:
        logger.exception(f"Worker failed for {item_id} (status={status}): {e}")
        return False


async def _worker_iteration(pool: asyncpg.Pool) -> int:
    """Process one batch. Returns the number of rows handled (0 = nothing to do).

    Only rebuilds the feature registry when at least one 'unprocessed' row
    succeeded — that's the only path that can add new room_instances /
    features, so 'image_only_processed' rows don't trigger a rebuild.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await claim_pending_batch(conn, limit=WORKER_BATCH_SIZE)
    if not rows:
        return 0

    results = await asyncio.gather(
        *[_process_one_row(pool, row) for row in rows],
        return_exceptions=True,
    )

    features_may_have_changed = any(
        ok is True and row["status"] == "unprocessed"
        for ok, row in zip(results, rows)
    )
    if features_may_have_changed:
        # The worker lives in a separate process and has no in-memory registry
        # of its own. It just tells the web tier (which DOES have a registry)
        # to rebuild via Postgres NOTIFY. Any other process LISTENing on the
        # `feature_change` channel will pick this up.
        try:
            async with pool.acquire() as conn:
                await conn.execute("NOTIFY feature_change")
        except Exception as e:
            logger.warning(f"Worker: NOTIFY feature_change failed: {e}")

    return len(rows)


async def run_worker_forever() -> None:
    """Top-level worker loop. Started in the FastAPI lifespan."""
    logger.info("Background worker starting")
    while True:
        try:
            pool = await get_pool()
            n = await _worker_iteration(pool)
            if n == 0:
                await asyncio.sleep(WORKER_IDLE_SLEEP_SECONDS)
        except asyncio.CancelledError:
            logger.info("Background worker received cancellation; exiting")
            raise
        except Exception as e:
            logger.exception(f"Background worker iteration crashed: {e}")
            await asyncio.sleep(WORKER_ERROR_SLEEP_SECONDS)
