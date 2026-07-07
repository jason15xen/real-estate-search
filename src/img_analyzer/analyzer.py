"""GPT-5.1 vision photo analyzer: extracts room type, color, and features."""

import asyncio
import json
import logging
from pathlib import Path

from config.settings import settings
from src.llm_client import get_async_client
from src.img_analyzer.models import Photo, PhotoResult

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompt"


ALLOWED_COLORS = {
    "white", "black", "gray", "brown", "beige", "blue", "green",
    "red", "yellow", "purple", "pink", "orange", "gold",
}


def _normalize_color(raw: str | None) -> str | None:
    """Coerce LLM color output to the 13-color palette, else None."""
    if not raw or not isinstance(raw, str):
        return None
    cleaned = raw.strip().lower()
    if cleaned in ALLOWED_COLORS:
        return cleaned
    if cleaned in {"", "unknown", "n/a", "none", "null"}:
        return None
    # Fallback synonym mapping
    SYNONYMS = {
        "ivory": "white", "cream": "white", "eggshell": "white", "off-white": "white",
        "navy": "blue", "teal": "blue", "turquoise": "blue",
        "tan": "beige", "khaki": "beige", "taupe": "beige", "sand": "beige",
        "charcoal": "gray", "silver": "gray", "grey": "gray",
        "wood": "brown", "wood-tone": "brown", "walnut": "brown", "oak": "brown", "mahogany": "brown",
        "sage": "green", "olive": "green", "forest": "green",
        "burgundy": "red", "maroon": "red", "coral": "red",
        "mustard": "yellow",
        "lavender": "purple", "violet": "purple",
        "salmon": "pink", "rose": "pink",
    }
    return SYNONYMS.get(cleaned)


def _load_prompt() -> str:
    return (PROMPT_DIR / "prompt.txt").read_text(encoding="utf-8")


def _load_features() -> str:
    return (PROMPT_DIR / "feature.txt").read_text(encoding="utf-8")


def build_vision_system_prompt() -> str:
    """Full vision system prompt (base + feature list); shared by the sync and Batch API paths."""
    return (
        f"{_load_prompt()}\n\n"
        f"# Known Real Estate Features (use these as reference for keyword extraction):\n"
        f"{_load_features()}"
    )


_GROUP_OUTPUT_INSTRUCTIONS = """

# GROUPED IMAGE MODE (batch analysis)
You will be shown {n} images of ONE property, each preceded by a text label "IMAGE k"
(k = 1..{n}). Analyze EVERY image independently, exactly as specified above.

Return STRICT JSON of this exact shape — an object with a single "results" array
containing EXACTLY {n} entries, one per image, in ascending image order:

{{"results": [
  {{"image": 1, "RoomType": "...", "Color": "...", "Features": ["...", "..."]}},
  {{"image": 2, "RoomType": "...", "Color": "...", "Features": ["..."]}}
]}}

Rules (CRITICAL for correct matching):
- Every entry MUST carry the "image" number copied from that image's label.
- EXACTLY one entry per image: no image skipped, none duplicated, no extras.
- Never merge observations across images; each entry describes only its own image.
No commentary, no markdown fences."""


def build_grouped_system_prompt(n_images: int) -> str:
    """Vision prompt for grouped (multi-image) batch requests: base prompt + feature
    list ONCE, plus strict per-image output-matching rules."""
    return build_vision_system_prompt() + _GROUP_OUTPUT_INSTRUCTIONS.format(n=n_images)


def parse_group_output(raw: str, n_images: int) -> list[dict] | None:
    """Parse+validate a grouped reply. Returns a list of n result dicts ORDERED BY IMAGE
    INDEX ({room_type, color, features}), or None if alignment cannot be PROVEN:
    non-JSON, wrong entry count, or any missing/duplicated image index. Per-entry field
    sloppiness is tolerated (coerced like the single-image parser); index structure is
    not — a group that fails here must be retried, never guessed."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))
    try:
        data = json.loads(raw)
        entries = data.get("results")
        if not isinstance(entries, list) or len(entries) != n_images:
            return None
        by_index: dict[int, dict] = {}
        for e in entries:
            idx = int(e.get("image"))
            if idx in by_index or not (1 <= idx <= n_images):
                return None  # duplicate or out-of-range index
            features = e.get("Features")
            by_index[idx] = {
                "room_type": str(e.get("RoomType") or "Unknown"),
                "color": _normalize_color(e.get("Color") if isinstance(e.get("Color"), str) else None),
                "features": [str(f) for f in features] if isinstance(features, list) else [],
            }
        if len(by_index) != n_images:
            return None
        return [by_index[i] for i in range(1, n_images + 1)]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError):
        return None


def parse_vision_content(raw: str, url: str) -> PhotoResult:
    """Parse a vision JSON reply into a PhotoResult (Unknown stub on any parse failure); shared by the sync and Batch API paths."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        raw = "\n".join(lines)
    try:
        data = json.loads(raw)
        return PhotoResult(
            photo_url=url,
            room_type=data.get("RoomType") or "Unknown",
            color=_normalize_color(data.get("Color")),
            features=data.get("Features") or [],
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
        # AttributeError: json.loads('null'/'"str"') yields a non-dict → .get() blows up.
        logger.error(f"Failed to parse vision response for {url}: {e}")
        return PhotoResult(photo_url=url, room_type="Unknown", color=None, features=[])


def _pick_jpeg_url(photo: Photo) -> str | None:
    """Highest-resolution JPEG URL from a photo."""
    jpegs = photo.mixedSources.jpeg
    if not jpegs:
        return None
    return sorted(jpegs, key=lambda j: j.width, reverse=True)[0].url


async def analyze_single_image(url: str, system_prompt: str) -> PhotoResult:
    """Analyze one image URL via GPT-5.1 vision."""
    client = get_async_client()

    response = await client.chat.completions.create(
        model=settings.openai_model,
        max_completion_tokens=1000,
        response_format={"type": "json_object"},
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

    raw = response.choices[0].message.content or ""
    return parse_vision_content(raw, url)


async def analyze_photos(
    property_id: str,
    photos: list[Photo],
    concurrency: int = 5,
) -> list[PhotoResult]:
    """Analyze all property photos via GPT-5.1 vision, semaphore-limited."""
    system_prompt = build_vision_system_prompt()

    # (index, url) pairs to preserve originalPhotos ordering
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
                return (idx, PhotoResult(photo_url=url, room_type="Unknown", color=None, features=[]))

    pairs = await asyncio.gather(*[_limited(i, u) for i, u in indexed_urls])
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    return [r for _, r in pairs_sorted]


def inject_features(raw_data: list[dict], results_map: dict[str, list[PhotoResult]]) -> list[dict]:
    """Inject RoomType/Color/Features into each originalPhotos entry, matched by photo URL (not position)."""
    for prop in raw_data:
        prop_id = prop.get("Id", "")
        by_url = {r.photo_url: r for r in results_map.get(prop_id, [])}
        photos = prop.get("ZillowPropertyRecord", {}).get("originalPhotos", [])

        for photo in photos:
            jpegs = photo.get("mixedSources", {}).get("jpeg", [])
            if not jpegs:
                continue
            best = max(jpegs, key=lambda j: j.get("width", 0))
            pr = by_url.get(best.get("url") or "")
            if pr:
                photo["RoomType"] = pr.room_type
                photo["Color"] = pr.color
                photo["Features"] = pr.features

    return raw_data
