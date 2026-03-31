"""
Image Analyzer Router — POST /process endpoint.

Accepts a JSON file upload (same format as data.json),
sends images to GPT-5.1 vision for feature extraction,
injects features back into each photo object, and saves
the enriched JSON to src/processed/data.json.
"""

import json
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.img_analyzer.analyzer import (
    analyze_photos,
    inject_features,
    save_processed,
)
from src.img_analyzer.db_ingest import ingest_processed_data
from src.img_analyzer.models import (
    PropertyItem,
    PropertyResult,
    ProcessResponse,
    SaveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Analyzer"])


@router.post("/process", response_model=ProcessResponse)
async def process_property_images(file: UploadFile = File(...)):
    """
    Upload a JSON file (same format as data.json) to extract
    room types and features from property photos using GPT-5.1 vision.

    Features are injected into each photo and saved to src/processed/data.json.
    """
    content = await file.read()
    raw_data: list[dict] = json.loads(content)

    # Parse into models for type-safe processing
    properties = [PropertyItem(**item) for item in raw_data]

    logger.info(f"Processing {len(properties)} properties from '{file.filename}'")

    all_results: list[PropertyResult] = []
    results_map: dict[str, list] = {}

    for prop in properties:
        photos = prop.ZillowPropertyRecord.originalPhotos
        logger.info(f"Property {prop.Id}: {len(photos)} photos")

        rooms = await analyze_photos(
            property_id=prop.Id,
            photos=photos,
        )

        results_map[prop.Id] = rooms

        all_results.append(
            PropertyResult(
                id=prop.Id,
                total_photos=len(photos),
                processed_photos=len(rooms),
                rooms=rooms,
            )
        )

    # Inject features into original data and save
    enriched = inject_features(raw_data, results_map)
    save_processed(enriched)

    return ProcessResponse(
        total_properties=len(all_results),
        results=all_results,
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
