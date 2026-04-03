"""
Image Analyzer — Sends property photos to GPT-5.1 vision to extract
room types and features for real estate search indexing.

Results are injected back into the original JSON data and saved
to src/processed/data.json.
"""

import asyncio
import json
import logging
from pathlib import Path

from config.settings import settings
from src.llm_client import get_async_client
from src.img_analyzer.models import Photo, PhotoResult

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompt"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"


def _load_prompt() -> str:
    return (PROMPT_DIR / "prompt.txt").read_text(encoding="utf-8")


def _load_features() -> str:
    return (PROMPT_DIR / "feature.txt").read_text(encoding="utf-8")


def _pick_jpeg_url(photo: Photo) -> str | None:
    """Pick the highest resolution JPEG URL from a photo."""
    jpegs = photo.mixedSources.jpeg
    if not jpegs:
        return None
    return sorted(jpegs, key=lambda j: j.width, reverse=True)[0].url


async def analyze_single_image(url: str, system_prompt: str) -> PhotoResult:
    """Send one image URL to GPT-5.1 vision and parse the response."""
    client = get_async_client()

    response = await client.chat.completions.create(
        model=settings.azure_openai_deployment,
        max_completion_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": "high"},
                    },
                ],
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
        return PhotoResult(
            photo_url=url,
            room_type=data.get("RoomType", "Unknown"),
            features=data.get("Features", []),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse vision response for {url}: {e}")
        return PhotoResult(photo_url=url, room_type="Unknown", features=[])


async def analyze_photos(
    property_id: str,
    photos: list[Photo],
    concurrency: int = 5,
) -> list[PhotoResult]:
    """
    Analyze all photos for a property using GPT-5.1 vision.
    Uses a semaphore to limit concurrent API calls.
    """
    base_prompt = _load_prompt()
    feature_list = _load_features()
    system_prompt = (
        f"{base_prompt}\n\n"
        f"# Known Real Estate Features (use these as reference for keyword extraction):\n"
        f"{feature_list}"
    )

    # Build (index, url) pairs to preserve ordering with originalPhotos
    indexed_urls: list[tuple[int, str]] = []
    for i, photo in enumerate(photos):
        url = _pick_jpeg_url(photo)
        if url:
            indexed_urls.append((i, url))

    if not indexed_urls:
        logger.warning(f"No JPEG URLs found for property {property_id}")
        return []

    logger.info(f"Analyzing {len(indexed_urls)} photos for property {property_id}")

    semaphore = asyncio.Semaphore(concurrency)

    async def _limited(idx: int, url: str) -> tuple[int, PhotoResult]:
        async with semaphore:
            try:
                result = await analyze_single_image(url, system_prompt)
                return (idx, result)
            except Exception as e:
                logger.error(f"Vision API error for {url}: {e}")
                return (idx, PhotoResult(photo_url=url, room_type="Unknown", features=[]))

    pairs = await asyncio.gather(*[_limited(i, u) for i, u in indexed_urls])
    # Sort by original index
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    return [r for _, r in pairs_sorted]


def inject_features(raw_data: list[dict], results_map: dict[str, list[PhotoResult]]) -> list[dict]:
    """
    Inject extracted features into the original JSON data.
    Each photo in originalPhotos gets RoomType and Features fields added.
    """
    for prop in raw_data:
        prop_id = prop.get("Id", "")
        photo_results = results_map.get(prop_id, [])
        photos = (
            prop.get("ZillowPropertyRecord", {}).get("originalPhotos", [])
        )

        result_idx = 0
        for photo in photos:
            # Match by checking if this photo had a JPEG URL we analyzed
            jpegs = photo.get("mixedSources", {}).get("jpeg", [])
            if jpegs:
                if result_idx < len(photo_results):
                    pr = photo_results[result_idx]
                    photo["RoomType"] = pr.room_type
                    photo["Features"] = pr.features
                result_idx += 1  # Always increment for photos with JEPGs

    return raw_data


def save_processed(data: list[dict]) -> Path:
    """Save the enriched JSON to src/processed/data.json, merging with existing data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "data.json"

    existing: dict[str, dict] = {}
    if output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text(encoding="utf-8"))
            existing = {item["Id"]: item for item in existing_data if "Id" in item}
        except json.JSONDecodeError:
            logger.warning("Could not parse existing data.json; starting fresh")

    for item in data:
        existing[item["Id"]] = item

    merged = list(existing.values())

    output_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved processed data → {output_path} ({len(merged)} total, {len(data)} upserted)")
    return output_path
