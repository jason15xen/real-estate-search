"""
Centralized Azure OpenAI client factory.

All LLM calls go through this module so the Azure endpoint + API key
configuration is managed in one place.
"""

from openai import AsyncAzureOpenAI

from config.settings import settings


def get_async_client() -> AsyncAzureOpenAI:
    """Client for Azure OpenAI — used for query parsing and geocoding."""
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
