"""
Image Analyzer Router

This module exposes the single ingest endpoint POST /process plus a small
read-only endpoint to inspect the raw staging table.

Ingest flow:
    POST /process   ──►  raw_properties  ──►  (background worker)  ──►  primary tables
                         ^                    ^
                         |                    └── continuous async worker
                         └── status: unprocessed | image_only_processed | processed

The previous endpoints (POST /properties, PUT /properties,
POST /saveprocesseddata, GET /job/*, GET /jobs) are intentionally removed —
all add/update traffic flows through POST /process.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body

from src.data.database import get_pool
from src.img_analyzer.models import PropertyInput
from src.img_analyzer.raw_db import get_status_counts, upsert_raw_property

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Analyzer"])


@router.post("/process", status_code=202)
async def process_properties(items: list[PropertyInput] = Body(...)):
    """Single ingest endpoint — add OR update properties in bulk.

    Body: JSON array of `{id, data}` objects where `data` is the full Zillow
    property record (including `originalPhotos`, `schools`, `resoFacts`, etc.).

    Each item lands in the raw_properties staging table with a status that
    tells the background worker how much work to do:

        ┌──────────────────────────────────────────────────────────────┐
        │ id is new                  → status = unprocessed            │
        │ id exists, photos changed  → status = unprocessed            │
        │ id exists, photos unchanged → status = image_only_processed  │
        └──────────────────────────────────────────────────────────────┘

    The endpoint returns immediately with summary counts. The worker handles
    Vision API analysis and writes to the primary search tables in the
    background. Poll GET /process/status if you want to see the queue depth.
    """
    pool = await get_pool()
    counts = {"unprocessed_new": 0, "unprocessed_changed": 0, "image_only_processed": 0, "failed": 0}

    async with pool.acquire() as conn:
        for it in items:
            try:
                # Distinguish "new" vs "changed" for the response by checking
                # existence before the upsert (best-effort, tracked separately
                # from the actual status assignment which lives in raw_db).
                exists = await conn.fetchval(
                    "SELECT 1 FROM raw_properties WHERE id = $1", str(it.id)
                )
                status = await upsert_raw_property(conn, str(it.id), it.data or {})
                if status == "image_only_processed":
                    counts["image_only_processed"] += 1
                elif status == "unprocessed":
                    if exists:
                        counts["unprocessed_changed"] += 1
                    else:
                        counts["unprocessed_new"] += 1
            except Exception as e:
                logger.exception(f"/process failed for id={it.id}: {e}")
                counts["failed"] += 1

    return {
        "total": len(items),
        "added": counts["unprocessed_new"],
        "updated_with_photo_changes": counts["unprocessed_changed"],
        "updated_without_photo_changes": counts["image_only_processed"],
        "failed": counts["failed"],
        "message": "Ingested into raw staging; background worker will sync to search index.",
    }


@router.get("/process/status")
async def process_status():
    """Quick view of the raw staging queue: how many rows are pending vs done."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts = await get_status_counts(conn)
    return {
        "unprocessed": counts.get("unprocessed", 0),
        "image_only_processed": counts.get("image_only_processed", 0),
        "processed": counts.get("processed", 0),
        "total": sum(counts.values()),
    }
