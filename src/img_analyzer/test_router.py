"""Vision A/B test router (POST /test/compare-vision): runs one prompt+image through primary and secondary OpenAI models, returning parsed results plus latency/usage."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import settings
from src.img_analyzer.analyzer import _load_features, _load_prompt, _normalize_color
from src.img_analyzer.models import PhotoResult
from src.llm_client import get_async_client, get_test_async_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Vision A/B Test"])


class CompareVisionRequest(BaseModel):
    photo_url: str


class ModelUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0


class ModelOutcome(BaseModel):
    deployment: str
    model: str
    latency_seconds: float
    usage: ModelUsage
    result: PhotoResult
    parse_ok: bool


class CompareVisionResponse(BaseModel):
    photo_url: str
    primary: ModelOutcome     # OPENAI_MODEL (gpt-5.1)
    secondary: ModelOutcome   # OPENAI_TEST_MODEL (gpt-5.4-mini)


def _build_system_prompt() -> str:
    base = _load_prompt()
    features = _load_features()
    return (
        f"{base}\n\n"
        "# Known Real Estate Features (use these as reference for keyword extraction):\n"
        f"{features}"
    )


async def _call_one(
    client,
    deployment: str,
    model_name: str,
    system_prompt: str,
    url: str,
) -> ModelOutcome:
    """One Vision call against one deployment. Captures latency + usage."""
    t0 = time.monotonic()
    response = await client.chat.completions.create(
        model=deployment,
        max_completion_tokens=1000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                ],
            },
        ],
    )
    elapsed = time.monotonic() - t0

    u = response.usage
    cached = 0
    if u and hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
        cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
    usage = ModelUsage(
        prompt_tokens=u.prompt_tokens if u else 0,
        completion_tokens=u.completion_tokens if u else 0,
        total_tokens=u.total_tokens if u else 0,
        cached_input_tokens=cached,
    )

    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    parse_ok = True
    try:
        data = json.loads(raw)
        result = PhotoResult(
            photo_url=url,
            room_type=data.get("RoomType", "Unknown"),
            color=_normalize_color(data.get("Color")),
            features=data.get("Features", []),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        parse_ok = False
        logger.warning(f"{deployment}: failed to parse response for {url[:80]}: {e}")
        result = PhotoResult(photo_url=url, room_type="Unknown", color=None, features=[])

    return ModelOutcome(
        deployment=deployment,
        model=model_name,
        latency_seconds=round(elapsed, 3),
        usage=usage,
        result=result,
        parse_ok=parse_ok,
    )


@router.post("/test/compare-vision", response_model=CompareVisionResponse)
async def compare_vision(req: CompareVisionRequest):
    """A/B-test primary and secondary deployments in parallel on the same image and prompt; returns latency and token usage."""
    if not settings.openai_test_model:
        raise HTTPException(
            status_code=503,
            detail="Secondary model not configured. Set OPENAI_TEST_MODEL in .env.",
        )

    url = req.photo_url
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="photo_url must be http(s)://")

    system_prompt = _build_system_prompt()
    primary_client = get_async_client()
    secondary_client = get_test_async_client()

    try:
        primary, secondary = await asyncio.gather(
            _call_one(
                primary_client,
                settings.openai_model,
                settings.openai_model,
                system_prompt,
                url,
            ),
            _call_one(
                secondary_client,
                settings.openai_test_model,
                settings.openai_test_model,
                system_prompt,
                url,
            ),
        )
    except Exception as e:
        logger.exception(f"compare-vision failed for {url[:80]}: {e}")
        raise HTTPException(status_code=502, detail=f"Vision API error: {e}")

    return CompareVisionResponse(
        photo_url=url,
        primary=primary,
        secondary=secondary,
    )
