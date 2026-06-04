"""Background worker: drains raw_properties → primary tables.

Runs in its own process (`python -m src.img_analyzer.worker_main`), loops
forever processing WORKER_BATCH_SIZE rows in parallel via asyncio.gather.
Claims rows with status 'unprocessed', 'image_only_processed', or
'partial_image_only_processed' and marks them 'processed'. On error a row's
status is left untouched so the next pass retries it.
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
    _canonical_photo_url,
    _extract_property_fields,
    _extract_schools,
    _get_room_counts,
    _insert_children,
    refresh_property_room_counts,
    update_property_scalars,
    update_property_with_children,
)
from src.img_analyzer.models import Photo, PropertyItem
from src.img_analyzer.raw_db import (
    claim_pending_batch,
    extract_photo_urls_from_data,
    mark_processed,
    primary_photo_urls,
)

logger = logging.getLogger(__name__)

WORKER_BATCH_SIZE = 3            # rows processed in parallel per iteration
WORKER_IDLE_SLEEP_SECONDS = 5    # sleep when no work
WORKER_ERROR_SLEEP_SECONDS = 10  # sleep after an errored iteration


async def _process_unprocessed(
    pool: asyncpg.Pool, item_id: str, data: dict, expected_updated_at,
) -> bool:
    """Full sync: Vision API → primary tables (INSERT or full re-INSERT of children).

    Returns False if a concurrent /process write raced us (status stays pending
    so the next iteration re-processes with fresh data).
    """
    item = {"Id": item_id, "ZillowPropertyRecord": data}

    p = PropertyItem(**item)  # raises on broken structure; caller logs
    photos = p.ZillowPropertyRecord.originalPhotos

    if photos:
        results = await analyze_photos(property_id=p.Id, photos=photos)
        inject_features([item], {p.Id: results})

    async with pool.acquire() as conn:
        # Transaction: child rebuild is DELETE-then-INSERT, so a mid-write crash
        # must not leave children deleted-but-not-reinserted.
        async with conn.transaction():
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
    """Metadata-only sync: scalars + Bedroom/Bathroom + schools; rooms untouched.

    Uses update_property_scalars (not update_property_metadata) to avoid the
    count-zeroing bug: most records have empty resoFacts.rooms and no Vision
    tags this round, so update_property_metadata would zero kitchen/living/
    dining/garage. Falls back to full unprocessed path if primary row missing.
    """
    async with pool.acquire() as conn:
        existing_id = await conn.fetchval(
            "SELECT id FROM properties WHERE guid = $1", item_id
        )
    if existing_id is None:
        return await _process_unprocessed(pool, item_id, data, expected_updated_at)

    item = {"Id": item_id, "ZillowPropertyRecord": data}
    async with pool.acquire() as conn:
        async with conn.transaction():
            await update_property_scalars(conn, existing_id, item)
            # Path-independent recompute of Kitchen/Living/Dining/Garage counts.
            await refresh_property_room_counts(conn, existing_id, item)
            await conn.execute(
                "DELETE FROM property_schools WHERE property_id = $1", existing_id
            )
            for s in _extract_schools(item):
                await conn.execute(
                    """
                    INSERT INTO property_schools (
                        property_id, school_name, rating, grades, distance_miles, link
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    existing_id, s["name"], s["rating"], s["grades"],
                    s["distance"], s["link"],
                )
        return await mark_processed(conn, item_id, expected_updated_at)


async def _process_partial(
    pool: asyncpg.Pool,
    item_id: str,
    data: dict,
    expected_updated_at,
) -> bool:
    """Partial image sync: recompute the photo diff against LIVE primary state,
    re-analyze only the still-needed added photos, delete room_instances for
    removed photos, refresh metadata + schools.

    Recomputing (vs. a stored diff) makes this idempotent: if an earlier attempt
    partially applied before losing its mark_processed race, the re-claim only
    does what's still missing — no duplicate room_instances. Empty diff →
    metadata-only refresh. Missing primary row → full unprocessed path.
    """
    item = {"Id": item_id, "ZillowPropertyRecord": data}

    # Step 1: existing_id + recompute diff against live primary state.
    async with pool.acquire() as conn:
        existing_id = await conn.fetchval(
            "SELECT id FROM properties WHERE guid = $1", item_id
        )
        if existing_id is None:
            primary_urls: set[str] = set()
        else:
            primary_urls = await primary_photo_urls(conn, item_id)

    if existing_id is None:
        # Defensive: partial implies primary exists; if not, full-process.
        return await _process_unprocessed(pool, item_id, data, expected_updated_at)

    desired_urls: set[str] = set(extract_photo_urls_from_data(data))
    actual_added = desired_urls - primary_urls
    actual_removed = sorted(primary_urls - desired_urls)

    # Step 2: locate Photo dicts for actual_added (dedupe by URL).
    photos_to_analyze: list[dict] = []
    seen_urls: set[str] = set()
    for photo_dict in (data.get("originalPhotos") or []):
        url = _canonical_photo_url(photo_dict)
        if url and url in actual_added and url not in seen_urls:
            seen_urls.add(url)
            photos_to_analyze.append(photo_dict)

    # Step 3: Vision API on just the freshly-needed photos.
    photo_results = []
    if photos_to_analyze:
        photo_models = [Photo(**ph) for ph in photos_to_analyze]
        photo_results = await analyze_photos(
            property_id=item_id, photos=photo_models, concurrency=5
        )

    if not photos_to_analyze and not actual_removed:
        # Empty diff (earlier attempt already applied it): metadata-only refresh.
        # update_property_scalars for the same count-zeroing reason as image_only.
        logger.info(
            f"Worker {item_id}: partial diff empty after recompute (idempotent skip)"
        )
        async with pool.acquire() as conn:
            async with conn.transaction():
                await update_property_scalars(conn, existing_id, item)
                await refresh_property_room_counts(conn, existing_id, item)
                await conn.execute(
                    "DELETE FROM property_schools WHERE property_id = $1", existing_id
                )
                for s in _extract_schools(item):
                    await conn.execute(
                        """
                        INSERT INTO property_schools (
                            property_id, school_name, rating, grades, distance_miles, link
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        existing_id, s["name"], s["rating"], s["grades"],
                        s["distance"], s["link"],
                    )
            return await mark_processed(conn, item_id, expected_updated_at)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Drop room_instances for removed photos.
            if actual_removed:
                await conn.execute(
                    """
                    DELETE FROM room_instances
                    WHERE property_id = $1 AND photo_url = ANY($2)
                    """,
                    existing_id,
                    actual_removed,
                )

            # 2. Insert room_instances for added photos, grouped by room_type.
            #    Vision-failed photos still get a stub Unknown instance so their
            #    URL is claimed — else /process re-flags them, burning quota in
            #    an infinite retry loop.
            new_rows: dict[str, list[dict]] = {}
            for photo, result in zip(photos_to_analyze, photo_results):
                url = _canonical_photo_url(photo)
                if not url:
                    continue  # no JPEG URL — untrackable
                if result.room_type and result.room_type != "Unknown" and result.features:
                    new_rows.setdefault(result.room_type, []).append({
                        "photo_url": url,
                        "features": list(result.features),
                        "color": result.color,
                    })
                else:
                    new_rows.setdefault("Unknown", []).append({
                        "photo_url": url,
                        "features": [],
                        "color": None,
                    })

            for room_type, instances in new_rows.items():
                # Find or create the rooms row for (property, room_type).
                room_id = await conn.fetchval(
                    "SELECT id FROM rooms WHERE property_id = $1 AND room_type = $2",
                    existing_id, room_type,
                )
                if room_id is None:
                    room_id = await conn.fetchval(
                        """
                        INSERT INTO rooms (property_id, room_type, count)
                        VALUES ($1, $2, 0) RETURNING id
                        """,
                        existing_id, room_type,
                    )

                # Append after the highest existing instance_index.
                max_idx = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX(instance_index), -1)
                    FROM room_instances
                    WHERE property_id = $1 AND room_type = $2
                    """,
                    existing_id, room_type,
                )
                for offset, inst in enumerate(instances):
                    features_text = ", ".join(inst["features"])
                    await conn.execute(
                        """
                        INSERT INTO room_instances (
                            room_id, property_id, room_type, instance_index,
                            features, features_text, color, photo_url
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        room_id, existing_id, room_type,
                        max_idx + 1 + offset,
                        inst["features"], features_text,
                        inst["color"], inst["photo_url"],
                    )

            # 3. Resync rooms.count from room_instances; drop now-empty rooms.
            await conn.execute(
                """
                UPDATE rooms
                SET count = (
                    SELECT COUNT(*) FROM room_instances ri
                    WHERE ri.property_id = rooms.property_id
                      AND ri.room_type = rooms.room_type
                )
                WHERE property_id = $1
                """,
                existing_id,
            )
            await conn.execute(
                "DELETE FROM rooms WHERE property_id = $1 AND count = 0",
                existing_id,
            )

            # 4. Refresh scalars + Bed/Bath and recompute Kitchen/Living/Dining/
            #    Garage (path-independent, same algorithm as full-process).
            await update_property_scalars(conn, existing_id, item)
            await refresh_property_room_counts(conn, existing_id, item)

            # 5. Refresh schools.
            await conn.execute(
                "DELETE FROM property_schools WHERE property_id = $1", existing_id
            )
            for s in _extract_schools(item):
                await conn.execute(
                    """
                    INSERT INTO property_schools (
                        property_id, school_name, rating, grades, distance_miles, link
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    existing_id, s["name"], s["rating"], s["grades"],
                    s["distance"], s["link"],
                )

        # 6. Optimistic-concurrency mark_processed.
        return await mark_processed(conn, item_id, expected_updated_at)


async def _insert_primary_full(conn, item: dict) -> None:
    """INSERT a new property + all child rows. Used for new arrivals and the
    rare image_only_processed case where the primary row is missing.
    """
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    room_counts = _get_room_counts(
        record, {rt: len(insts) for rt, insts in rooms_from_photos.items()}
    )

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

    Returns False if raced or if processing raised; the row stays pending for
    the next iteration either way.
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
        elif status == "partial_image_only_processed":
            # No stored diff; _process_partial recomputes against primary.
            marked = await _process_partial(
                pool, item_id, raw_data, expected_updated_at
            )
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
    """Process one batch; returns rows handled (0 = nothing to do).

    Only triggers a feature-registry rebuild when an unprocessed or partial row
    succeeded — those are the only paths that add room_instances/features.
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
        ok is True
        and row["status"] in ("unprocessed", "partial_image_only_processed")
        for ok, row in zip(results, rows)
    )
    if features_may_have_changed:
        # Worker has no in-memory registry; signal the web tier to rebuild via
        # Postgres NOTIFY on the feature_change channel.
        try:
            async with pool.acquire() as conn:
                await conn.execute("NOTIFY feature_change")
        except Exception as e:
            logger.warning(f"Worker: NOTIFY feature_change failed: {e}")

    return len(rows)


async def run_worker_forever() -> None:
    """Top-level worker loop. Started by worker_main in the worker process."""
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
