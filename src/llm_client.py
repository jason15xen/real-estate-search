"""Centralized LLM client factory — one shared AsyncOpenAI client (single OPENAI_API_KEY) for vision, geocoding, query parsing, feature filtering, and embeddings."""

from functools import lru_cache

from openai import AsyncOpenAI

from config.settings import settings


@lru_cache(maxsize=1)
def get_async_client() -> AsyncOpenAI:
    """Shared OpenAI client — vision, geocoding, chat, and embeddings all use it."""
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_query_client() -> AsyncOpenAI:
    """Client for /search query parsing + feature filtering (same OpenAI key)."""
    return get_async_client()


def get_test_async_client() -> AsyncOpenAI:
    """Client for the /test/compare-vision A/B endpoint (same key, different model)."""
    return get_async_client()


def get_embedding_client() -> AsyncOpenAI:
    """Client for embeddings (same OpenAI key)."""
    return get_async_client()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings; returns one 1536-dim vector per input in order, empty input → empty list."""
    if not texts:
        return []
    client = get_embedding_client()
    resp = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    # API returns results in request order; sort by index defensively.
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]
