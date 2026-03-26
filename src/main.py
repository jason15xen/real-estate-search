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

from fastapi import FastAPI
from pydantic import BaseModel

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.data.feature_registry import registry
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
    version="0.2.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    results: list[dict]
    understood_intent: str
    stats: dict


@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    pool = await get_pool()
    result = await search(request.query, pool)

    return SearchResponse(
        results=result["results"],
        understood_intent=result["parsed_query"].understood_intent,
        stats=result["stats"],
    )


@app.get("/health")
async def health():
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM properties")
    return {"status": "ok", "properties_in_db": count}
