"""Image Analyzer Router: ingest via POST /process into raw staging; background worker syncs to primary tables."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from pydantic import ValidationError

from src.data.database import get_pool
from src.img_analyzer.models import PropertyInput
from src.img_analyzer.raw_db import get_status_counts, upsert_raw_property

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Analyzer"])


async def _ingest_items(items: list[PropertyInput]) -> dict:
    """Shared ingest: upsert items into raw_properties with a worker status; returns summary counts immediately."""
    pool = await get_pool()
    counts = {
        "unprocessed_new": 0,
        "unprocessed_full_rebuild": 0,
        "partial": 0,
        "image_only_processed": 0,
        "failed": 0,
    }
    partial_diffs: list[dict] = []

    async with pool.acquire() as conn:
        for it in items:
            try:
                status, image_diff, existed = await upsert_raw_property(
                    conn, str(it.id), it.data or {}
                )
                if status == "image_only_processed":
                    counts["image_only_processed"] += 1
                elif status == "partial_image_only_processed":
                    counts["partial"] += 1
                    if image_diff:
                        partial_diffs.append({
                            "id": str(it.id),
                            "added": len(image_diff.get("added", [])),
                            "removed": len(image_diff.get("removed", [])),
                        })
                elif status == "unprocessed":
                    if existed:
                        counts["unprocessed_full_rebuild"] += 1
                    else:
                        counts["unprocessed_new"] += 1
            except Exception as e:
                logger.exception(f"/process failed for id={it.id}: {e}")
                counts["failed"] += 1

    return {
        "total": len(items),
        "added": counts["unprocessed_new"],
        "updated_with_all_photos_changed": counts["unprocessed_full_rebuild"],
        "updated_with_partial_photo_changes": counts["partial"],
        "updated_without_photo_changes": counts["image_only_processed"],
        "failed": counts["failed"],
        "partial_diffs": partial_diffs,
        "message": "Ingested into raw staging; background worker will sync to search index.",
    }


@router.post("/process", status_code=202)
async def process_properties(items: list[PropertyInput] = Body(...)):
    """Bulk add/update properties. Body: JSON array of {id, data}. Returns summary counts."""
    return await _ingest_items(items)


@router.post("/process/upload", status_code=202)
async def process_properties_upload(file: UploadFile = File(...)):
    """Like POST /process but payload is an uploaded JSON file (field `file`): array of {id, data}."""
    raw = await file.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Uploaded file is not valid JSON: {e}"
        )

    if not isinstance(payload, list):
        raise HTTPException(
            status_code=400,
            detail="Uploaded JSON must be an array of {id, data} objects.",
        )

    try:
        items = [PropertyInput.model_validate(obj) for obj in payload]
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))

    return await _ingest_items(items)


@router.get("/process/status")
async def process_status():
    """Raw staging queue counts: pending vs done."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts = await get_status_counts(conn)
    return {
        "unprocessed": counts.get("unprocessed", 0),
        "image_only_processed": counts.get("image_only_processed", 0),
        "partial_image_only_processed": counts.get("partial_image_only_processed", 0),
        "processed": counts.get("processed", 0),
        "total": sum(counts.values()),
    }
