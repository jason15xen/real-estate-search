"""Real Estate AI Search — FastAPI app.

Multi-phase pipeline returning only properties matching ALL criteria:
1. LLM parses query → structured criteria (synonyms → known features)
2. Hard filters (Postgres) 3. Proximity filters (PostGIS)
4. Feature matching on room_instances.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from starlette.responses import Response

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.data.feature_registry import registry
from src.img_analyzer.router import router as img_analyzer_router
from src.img_analyzer.test_router import router as vision_test_router
from src.models.search import ParsedQuery
from src.search.orchestrator import search

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    logger.info("Database pool initialized")

    logger.info("Building feature registry from database...")
    await registry.build_from_db(pool)
    logger.info(
        f"Registry ready: {len(registry.features)} features, "
        f"{len(registry.room_types)} room types"
    )

    # Web tier LISTENs for 'feature_change' to keep the in-memory registry fresh;
    # supervisor heartbeats + reconnects so a Postgres blip can't leave us deaf to NOTIFYs.
    supervisor = _ListenerSupervisor(pool)
    supervisor_task = asyncio.create_task(supervisor.run())
    logger.info("LISTEN feature_change supervisor started")

    yield

    await supervisor.stop()
    supervisor_task.cancel()
    try:
        await supervisor_task
    except asyncio.CancelledError:
        pass
    logger.info("Feature-change listener stopped")

    await close_pool()
    logger.info("Database pool closed")


async def _on_feature_change(_connection, _pid, channel, _payload):
    logger.info(f"Received NOTIFY {channel} → rebuilding feature registry")
    try:
        pool = await get_pool()
        await registry.build_from_db(pool)
        logger.info(
            f"Registry refreshed: {len(registry.features)} features, "
            f"{len(registry.room_types)} room types"
        )
    except Exception as e:
        logger.error(f"Registry rebuild after notify failed: {e}")


class _ListenerSupervisor:
    """Dedicated LISTEN connection that reconnects on failure.

    LISTEN state lives on the server session, so the pool can't auto-recover it.
    Probe every HEARTBEAT_SEC with SELECT 1; on error, re-acquire and re-LISTEN.
    """

    HEARTBEAT_SEC = 30
    RECONNECT_BACKOFF_SEC = 2

    def __init__(self, pool):
        self._pool = pool
        self._stopping = False

    async def stop(self) -> None:
        self._stopping = True

    async def run(self) -> None:
        while not self._stopping:
            conn = None
            try:
                conn = await self._pool.acquire()
                await conn.add_listener("feature_change", _on_feature_change)
                logger.info("LISTENing for feature_change notifications")
                while not self._stopping:
                    await asyncio.sleep(self.HEARTBEAT_SEC)
                    await conn.fetchval("SELECT 1")  # raises if session is gone
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"Listener connection lost ({type(e).__name__}: {e}); "
                    f"reconnecting in {self.RECONNECT_BACKOFF_SEC}s"
                )
            finally:
                if conn is not None:
                    try:
                        await conn.remove_listener("feature_change", _on_feature_change)
                    except Exception:
                        pass
                    try:
                        await self._pool.release(conn, discard=True)
                    except Exception:
                        pass
            if not self._stopping:
                await asyncio.sleep(self.RECONNECT_BACKOFF_SEC)


app = FastAPI(
    title="Real Estate AI Search",
    description="Accurate AI-powered real estate search — returns only properties that match ALL criteria.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Must be False with wildcard origin (CORS spec forbids the combo); API is token/cookie-free.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(img_analyzer_router)
app.include_router(vision_test_router)


LOG_DIR = Path(settings.log_dir)
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    # Don't abort startup/import if the log dir isn't writable (e.g. local dev,
    # tests). Request logging just no-ops — see _append_log_entry's guard.
    logger.warning(f"Could not create log dir {LOG_DIR}: {e}; request logging disabled")
MAX_BODY_LOG_BYTES = 200_000
_log_write_lock = asyncio.Lock()


def _normalize_bounds(body: dict) -> dict:
    """Coerce bounds values to floats for logging."""
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
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    return LOG_DIR / f"{year}-W{week:02d}.jsonl"


async def _append_log_entry(entry: dict) -> None:
    if not LOG_DIR.is_dir():
        return  # log dir unavailable (see startup warning) — skip silently
    # Append one JSON object per line — O(1) per request, no whole-file rewrite.
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    async with _log_write_lock:
        with _weekly_log_path().open("a", encoding="utf-8") as f:
            f.write(line)


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
            filters_dict = None  # empty {} = no filters

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
        logger.exception(f"Search failed for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail="Search failed due to an internal error.")


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
              AND room_type != 'Unknown'    -- hide internal claim stubs
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
