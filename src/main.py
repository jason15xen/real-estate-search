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

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.data.feature_registry import registry
from src.img_analyzer.router import router as img_analyzer_router
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


class Bounds(BaseModel):
    north: str
    south: str
    east: str
    west: str


class SearchRequest(BaseModel):
    query: str
    bounds: Bounds | None = None


class SearchResponse(BaseModel):
    query: str
    zillowProperties: list[str]


@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    try:
        pool = await get_pool()
        bounds_dict = request.bounds.model_dump() if request.bounds else None
        result = await search(request.query, pool, bounds=bounds_dict)

        # Extract GUIDs from results
        guids = [r["Id"] for r in result["results"]]

        return SearchResponse(
            query=request.query,
            zillowProperties=guids,
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
