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
from pydantic import BaseModel, field_validator
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


ALLOWED_HOME_TYPES = {"SINGLE_FAMILY", "CONDO", "TOWNHOUSE", "MANUFACTURED", "MULTI_FAMILY"}


class Filters(BaseModel):
    price_min: int | None = None
    price_max: int | None = None
    beds_min: int | None = None
    baths_min: int | None = None
    property_types: list[str] | None = None
    financing: list[str] | None = None
    sqft_min: int | None = None
    sqft_max: int | None = None
    year_from: int | None = None
    year_to: int | None = None

    @field_validator("property_types")
    @classmethod
    def _norm_property_types(cls, v):
        if v is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for item in v:
            up = str(item).strip().upper()
            if up not in ALLOWED_HOME_TYPES:
                raise ValueError(
                    f"Invalid property_type '{item}'. "
                    f"Allowed: {sorted(ALLOWED_HOME_TYPES)}"
                )
            if up not in seen:
                seen.add(up)
                normalized.append(up)
        return normalized

    @field_validator("financing")
    @classmethod
    def _norm_financing(cls, v):
        if v is None:
            return None
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            norm = "_".join(str(item).strip().lower().split())
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out


class SearchRequest(BaseModel):
    query: str
    bounds: Bounds | None = None
    filters: Filters | None = None
    debug: bool = False


class DebugInfo(BaseModel):
    parsed_query: ParsedQuery
    stats: dict
    filter_steps: list[dict]
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
        filters_dict = (
            request.filters.model_dump(exclude_none=True) if request.filters else None
        )
        if filters_dict is not None and not filters_dict:
            filters_dict = None  # treat empty {} as no filters

        result = await search(
            request.query,
            pool,
            bounds=bounds_dict,
            filters=filters_dict,
            debug=request.debug,
        )

        guids = result["guids"]

        debug_info = None
        if request.debug:
            debug_info = DebugInfo(
                parsed_query=result["parsed_query"],
                stats=result["stats"],
                filter_steps=result["filter_steps"] or [],
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


@app.get("/property/{guid}")
async def get_property(guid: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        prop_row = await conn.fetchrow("""
            SELECT
                id, guid, name, street, district, city, state, postal_code, country,
                ST_Y(geom::geometry) AS latitude,
                ST_X(geom::geometry) AS longitude,
                area_sqft, price_usd,
                bedroom_count, bathroom_count, kitchen_count,
                living_room_count, dining_room_count, garage_count,
                home_type, rent_estimate, year_built, lot_size_sqft, stories,
                description, created_at, updated_at
            FROM properties WHERE guid = $1
        """, guid)
        if not prop_row:
            raise HTTPException(status_code=404, detail=f"Property '{guid}' not found")

        room_rows = await conn.fetch("""
            SELECT room_type, instance_index, features
            FROM room_instances
            WHERE property_id = $1
            ORDER BY room_type, instance_index
        """, prop_row["id"])

        school_rows = await conn.fetch("""
            SELECT school_name, rating, grades, distance_miles, link
            FROM property_schools
            WHERE property_id = $1
            ORDER BY distance_miles
        """, prop_row["id"])

    rooms_map: dict[str, list[list[str]]] = {}
    all_features: set[str] = set()
    for r in room_rows:
        rooms_map.setdefault(r["room_type"], []).append(list(r["features"]))
        all_features.update(r["features"])

    rooms = [
        {
            "room_type": rt,
            "instance_count": len(instances),
            "instances": [{"features": feats} for feats in instances],
        }
        for rt, instances in rooms_map.items()
    ]

    schools = [
        {
            "name": s["school_name"],
            "rating": s["rating"],
            "grades": s["grades"],
            "distance_miles": float(s["distance_miles"]),
            "link": s["link"],
        }
        for s in school_rows
    ]

    return {
        "property_id": guid,
        "name": prop_row["name"],
        "address": {
            "street": prop_row["street"],
            "district": prop_row["district"],
            "city": prop_row["city"],
            "state": prop_row["state"],
            "postal_code": prop_row["postal_code"],
            "country": prop_row["country"],
        },
        "location": {
            "latitude": prop_row["latitude"],
            "longitude": prop_row["longitude"],
        },
        "area_sqft": prop_row["area_sqft"],
        "price_usd": prop_row["price_usd"],
        "room_counts": {
            "bedroom": prop_row["bedroom_count"],
            "bathroom": prop_row["bathroom_count"],
            "kitchen": prop_row["kitchen_count"],
            "living_room": prop_row["living_room_count"],
            "dining_room": prop_row["dining_room_count"],
            "garage": prop_row["garage_count"],
        },
        "attributes": {
            "home_type": prop_row["home_type"],
            "rent_estimate": prop_row["rent_estimate"],
            "year_built": prop_row["year_built"],
            "lot_size_sqft": prop_row["lot_size_sqft"],
            "stories": prop_row["stories"],
        },
        "description": prop_row["description"],
        "schools": schools,
        "total_features": len(all_features),
        "all_features": sorted(all_features),
        "rooms": rooms,
        "created_at": prop_row["created_at"].isoformat() if prop_row["created_at"] else None,
        "updated_at": prop_row["updated_at"].isoformat() if prop_row["updated_at"] else None,
    }
