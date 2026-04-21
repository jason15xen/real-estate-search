"""
Real Estate AI Search — FastAPI Application

An AI-powered search engine that returns ONLY properties matching ALL criteria.
The system uses a multi-phase pipeline:

  1. LLM parses query → structured criteria (maps synonyms to known feature names)
  2. Hard filters → PostgreSQL indexed queries
  3. Proximity filters → PostGIS spatial queries
  4. Feature matching → PostgreSQL text matching on room_instances

The result: accurate results, not just "similar" ones.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import Response

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.data.feature_registry import registry
from src.img_analyzer.router import router as img_analyzer_router
from src.models.search import ParsedQuery
from src.search.orchestrator import search

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database pool
    pool = await get_pool()
    logger.info("Database pool initialized")

    # Build feature registry from PostgreSQL
    logger.info("Building feature registry from database...")
    await registry.build_from_db(pool)
    logger.info(
        f"Registry ready: {len(registry.features)} features, "
        f"{len(registry.room_types)} room types"
    )

    yield

    await close_pool()
    logger.info("Database pool closed")


app = FastAPI(
    title="Real Estate AI Search",
    description="Accurate AI-powered real estate search — returns only properties that match ALL criteria.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(img_analyzer_router)


LOG_DIR = Path("/app/log")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MAX_BODY_LOG_BYTES = 200_000
_log_write_lock = asyncio.Lock()


def _normalize_bounds(body: dict) -> dict:
    """Ensure bounds values are logged as floats, even if the client sent strings."""
    bounds = body.get("bounds")
    if isinstance(bounds, dict):
        for key in ("north", "south", "east", "west"):
            if key in bounds:
                try:
                    bounds[key] = float(bounds[key])
                except (ValueError, TypeError):
                    pass
    return body


def _decode_body(raw: bytes, content_type: str) -> dict | str | None:
    if not raw:
        return None
    if len(raw) > MAX_BODY_LOG_BYTES:
        return {"_truncated": True, "size_bytes": len(raw)}
    if "application/json" in content_type:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = _normalize_bounds(parsed)
            return parsed
        except json.JSONDecodeError:
            pass
    if "multipart/form-data" in content_type:
        return {"_binary_upload": True, "size_bytes": len(raw)}
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"_binary": True, "size_bytes": len(raw)}


def _weekly_log_path() -> Path:
    year, week, _ = datetime.utcnow().isocalendar()
    return LOG_DIR / f"{year}-W{week:02d}.json"


async def _append_log_entry(entry: dict) -> None:
    async with _log_write_lock:
        path = _weekly_log_path()
        entries: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    entries = existing
            except json.JSONDecodeError:
                logger.warning(f"Log file {path.name} is corrupt, starting fresh")
        entries.append(entry)
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


@app.middleware("http")
async def log_request_response(request: Request, call_next):
    if request.url.path != "/search":
        return await call_next(request)

    req_body = await request.body()

    async def receive():
        return {"type": "http.request", "body": req_body, "more_body": False}

    request._receive = receive

    response = await call_next(request)

    resp_chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        resp_chunks.append(chunk)
    resp_body = b"".join(resp_chunks)

    new_response = Response(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

    try:
        entry = {
            "req": _decode_body(req_body, request.headers.get("content-type", "")),
            "res": _decode_body(resp_body, new_response.headers.get("content-type", "")),
        }
        await _append_log_entry(entry)
    except Exception as e:
        logger.error(f"Failed to write request log: {e}")

    return new_response


class Bounds(BaseModel):
    north: float
    south: float
    east: float
    west: float


class SearchRequest(BaseModel):
    query: str
    bounds: Bounds | None = None
    debug: bool = False


class DebugInfo(BaseModel):
    parsed_query: ParsedQuery
    stats: dict
    bounds_applied: bool


class SearchResponse(BaseModel):
    query: str
    zillowProperties: list[str]
    debug: DebugInfo | None = None


@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    try:
        pool = await get_pool()
        bounds_dict = request.bounds.model_dump() if request.bounds else None
        result = await search(request.query, pool, bounds=bounds_dict)

        guids = [r["Id"] for r in result["results"]]

        debug_info = None
        if request.debug:
            debug_info = DebugInfo(
                parsed_query=result["parsed_query"],
                stats=result["stats"],
                bounds_applied=bounds_dict is not None,
            )

        return SearchResponse(
            query=request.query,
            zillowProperties=guids,
            debug=debug_info,
        )
    except Exception as e:
        logger.error(f"Search failed for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health")
async def health():
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM properties")
    return {"status": "ok", "properties_in_db": count}
