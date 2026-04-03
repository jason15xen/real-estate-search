"""
Image Analyzer Router — POST /process, GET /job/{job_id}, POST /saveprocesseddata

/process runs in background and returns a job ID immediately.
Poll GET /job/{job_id} to check progress.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.img_analyzer.analyzer import (
    analyze_photos,
    inject_features,
    save_processed,
)
from src.img_analyzer.db_ingest import ingest_processed_data
from src.img_analyzer.job_manager import job_manager
from src.img_analyzer.models import (
    JobStatus,
    PropertyItem,
    PropertyResult,
    SaveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Analyzer"])


async def _process_in_background(
    job_id: str,
    raw_data: list[dict],
    properties: list[PropertyItem],
) -> None:
    """Background task: analyze photos, inject features, save results."""
    try:
        results_map: dict[str, list] = {}

        for prop in properties:
            photos = prop.ZillowPropertyRecord.originalPhotos
            logger.info(f"[Job {job_id}] Property {prop.Id}: {len(photos)} photos")

            rooms = await analyze_photos(
                property_id=prop.Id,
                photos=photos,
            )

            results_map[prop.Id] = rooms
            job_manager.update_progress(job_id)

        # Inject features into original data and save
        enriched = inject_features(raw_data, results_map)
        save_processed(enriched)

        job_manager.complete_job(job_id)
    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {e}")
        job_manager.fail_job(job_id, str(e))


@router.post("/process", response_model=JobStatus)
async def process_property_images(file: UploadFile = File(...)):
    """
    Upload a JSON file (same format as data.json) to extract
    room types and features from property photos using GPT-5.1 vision.

    Returns immediately with a job ID. Poll GET /job/{job_id} for progress.
    """
    content = await file.read()
    raw_data: list[dict] = json.loads(content)

    # Parse into models for type-safe processing
    properties = [PropertyItem(**item) for item in raw_data]

    logger.info(f"Starting background job for {len(properties)} properties from '{file.filename}'")

    # Create job and start background processing
    job = job_manager.create_job(total_properties=len(properties))
    asyncio.create_task(_process_in_background(job.job_id, raw_data, properties))

    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        total_properties=job.total_properties,
        processed_properties=job.processed_properties,
    )


@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Check the status of a background processing job.

    Poll this endpoint until status is "completed" or "failed".
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        total_properties=job.total_properties,
        processed_properties=job.processed_properties,
        error=job.error,
    )


@router.post("/saveprocesseddata", response_model=SaveResponse)
async def save_processed_data_to_db():
    """
    Read the processed data from src/processed/data.json, convert it
    to the PostgreSQL schema, and insert into the database.

    Run this AFTER /process has completed and saved the enriched data.
    The search pipeline (/search) will then use this data.
    """
    from src.data.database import get_pool
    from src.data.feature_registry import registry

    try:
        pool = await get_pool()
        stats = await ingest_processed_data(pool)

        # Rebuild feature registry so search uses the new data
        await registry.build_from_db(pool)
        logger.info(
            f"Registry rebuilt: {len(registry.features)} features, "
            f"{len(registry.room_types)} room types"
        )

        return SaveResponse(
            total_properties=stats["total_properties"],
            total_rooms=stats["total_rooms"],
            total_room_instances=stats["total_room_instances"],
            total_schools=stats["total_schools"],
            message="Data saved to database. Feature registry rebuilt. Search is ready.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
