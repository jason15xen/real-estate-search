"""Centralized LLM client factory — one cached client per provider to reuse
HTTPX/TCP pools. Azure GPT-5.1 for vision/geocoding; Claude Opus 4.7 (Azure AI
Foundry Anthropic endpoint) for /search query parsing."""

from functools import lru_cache

from anthropic import AsyncAnthropic
from openai import AsyncAzureOpenAI

from config.settings import settings


@lru_cache(maxsize=1)
def get_async_client() -> AsyncAzureOpenAI:
    """Shared Azure OpenAI client for vision and geocoding."""
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _azure_resource_base(endpoint: str) -> str:
    """Strip /openai[/v1] suffixes — AsyncAzureOpenAI builds the deployment URL
    itself from the resource base. The .env value sometimes carries the full
    /openai/v1/ path from the Azure Foundry portal."""
    e = (endpoint or "").rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if e.endswith(suffix):
            e = e[: -len(suffix)]
            break
    return e


@lru_cache(maxsize=1)
def get_test_async_client() -> AsyncAzureOpenAI:
    """Secondary Azure OpenAI client for /test/compare-vision A/B comparisons.
    Reads TEST_AZURE_OPENAI_* env vars; falls back to the primary api_version."""
    return AsyncAzureOpenAI(
        api_key=settings.test_azure_openai_api_key,
        azure_endpoint=_azure_resource_base(settings.test_azure_openai_endpoint),
        api_version=settings.azure_openai_api_version,
    )


def _claude_base_url() -> str:
    """Normalize endpoint into an Anthropic base_url. The SDK appends
    /v1/messages itself, so strip that suffix (and trailing slashes)."""
    endpoint = (settings.azure_openai_endpoint_for_query or "").rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return endpoint


@lru_cache(maxsize=1)
def get_claude_client() -> AsyncAnthropic:
    """Shared Claude client (Azure AI Foundry Anthropic endpoint) for /search
    query parsing (Opus 4.7)."""
    return AsyncAnthropic(
        api_key=settings.azure_openai_api_key_for_query,
        base_url=_claude_base_url(),
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> AsyncAzureOpenAI:
    """Dedicated Azure OpenAI client for the embedding deployment (own endpoint
    + key from EMBEDDING_SMALL_*)."""
    return AsyncAzureOpenAI(
        api_key=settings.embedding_small_api_key,
        azure_endpoint=_azure_resource_base(settings.embedding_small_endpoint),
        api_version=settings.azure_openai_api_version,
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings with the dedicated Azure embedding deployment.

    Returns one 1536-dim vector per input, in the same order. Empty input →
    empty list.
    """
    if not texts:
        return []
    client = get_embedding_client()
    resp = await client.embeddings.create(
        model=settings.embedding_small_deployment,
        input=texts,
    )
    # API returns results in request order; sort by index defensively.
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]
