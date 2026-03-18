"""
Real Estate AI Search — FastAPI Application

An AI-powered search engine that returns ONLY properties matching ALL criteria.
Unlike pure vector search (which returns similar results ranked by similarity),
this system uses a multi-phase pipeline:

  1. LLM parses query → structured criteria
  2. Hard filters → eliminate non-qualifying properties
  3. Proximity filters → distance-based elimination
  4. Vector search → semantic feature matching (handles synonyms)
  5. LLM validation → rejects false positives

The result: accurate results, not just "similar" ones.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from config.settings import settings
from src.data.loader import load_properties
from src.indexer.property_indexer import index_properties
from src.models.property import Property
from src.search.orchestrator import search

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory property store (loaded at startup)
_properties: list[Property] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _properties
    logger.info("Loading properties from mockup.json...")
    _properties = load_properties("mockup.json")
    logger.info(f"Loaded {len(_properties)} properties")

    logger.info("Indexing properties into vector database...")
    doc_count = index_properties(_properties)
    logger.info(f"Indexed {doc_count} documents")

    yield


app = FastAPI(
    title="Real Estate AI Search",
    description="Accurate AI-powered real estate search — returns only properties that match ALL criteria.",
    version="0.1.0",
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
    result = await search(request.query, _properties)

    # Return full property info matching the original data structure
    property_results = [prop.model_dump() for prop in result["results"]]

    return SearchResponse(
        results=property_results,
        understood_intent=result["parsed_query"].understood_intent,
        stats=result["stats"],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "properties_loaded": len(_properties)}
