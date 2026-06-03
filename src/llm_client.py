"""
Centralized LLM client factory.

Two providers are wired up — one cached client per provider so HTTPX sessions
and TCP pools are reused across requests:

  * Azure OpenAI (GPT-5.1) — image vision analysis and geocoding fallback
  * Claude Opus 4.7 via Azure AI Foundry's Anthropic-compatible endpoint —
    /search query parsing
"""

from functools import lru_cache

from anthropic import AsyncAnthropic
from openai import AsyncAzureOpenAI

from config.settings import settings


@lru_cache(maxsize=1)
def get_async_client() -> AsyncAzureOpenAI:
    """Shared Azure OpenAI client — used for image vision and geocoding."""
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _claude_base_url() -> str:
    """Normalize the configured endpoint into an Anthropic SDK base_url.

    The user's .env tends to carry the *full* endpoint
    (`https://…/anthropic/v1/messages`). The Anthropic SDK appends `/v1/messages`
    itself, so we strip that suffix here. Trailing slashes are also removed.
    """
    endpoint = (settings.azure_openai_endpoint_for_query or "").rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return endpoint


@lru_cache(maxsize=1)
def get_claude_client() -> AsyncAnthropic:
    """Shared Claude client pointed at the Azure AI Foundry Anthropic endpoint.

    Used exclusively for /search query parsing (Opus 4.7). Vision and geocoding
    stay on Azure GPT-5.1 via `get_async_client()`.
    """
    return AsyncAnthropic(
        api_key=settings.azure_openai_api_key_for_query,
        base_url=_claude_base_url(),
    )
