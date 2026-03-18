"""
Centralized Azure Anthropic client factory.

All LLM calls go through this module so the Azure endpoint + API key
configuration is managed in one place.

Two endpoints exist because Opus and Haiku are deployed on different
Azure services:
  - Opus:  services.ai.azure.com  (primary, used for validation)
  - Haiku: openai.azure.com       (fast, used for parsing/geocoding)
"""

import anthropic

from config.settings import settings


def get_async_client() -> anthropic.AsyncAnthropic:
    """Client for the primary (Opus) Azure endpoint."""
    return anthropic.AsyncAnthropic(
        api_key=settings.azure_anthropic_api_key,
        base_url=settings.azure_anthropic_endpoint,
    )


def get_async_client_fast() -> anthropic.AsyncAnthropic:
    """Client for the fast (Haiku) Azure endpoint."""
    return anthropic.AsyncAnthropic(
        api_key=settings.azure_anthropic_api_key,
        base_url=settings.azure_anthropic_endpoint_fast,
    )
