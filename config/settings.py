from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI (GPT-5.1) — vision analysis and geocoding.
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-5.1"
    azure_openai_model: str = "gpt-5.1"
    azure_openai_api_version: str = "2025-04-01-preview"

    # Claude (Azure AI Foundry Anthropic endpoint) — /search query parsing (Opus 4.7).
    azure_openai_api_key_for_query: str = ""
    azure_openai_endpoint_for_query: str = ""
    azure_openai_deployment_for_query: str = "claude-opus-4-7-dev"
    azure_openai_model_for_query: str = "claude-opus-4-7"

    # Embedding model — dedicated Azure deployment (own endpoint + key).
    # Used for feature-embedding retrieval in /search.
    embedding_small_endpoint: str = ""
    embedding_small_api_key: str = ""
    embedding_small_deployment: str = "text-embedding-3-small"
    embedding_small_model: str = "text-embedding-3-small"
    azure_openai_embedding_dim: int = 1536

    # /search feature-retrieval strategy:
    #   True  → embedding retrieval (top-K candidates) + Claude relevance filter
    #   False → legacy: dump every DB feature into the Claude #1 prompt
    # Flag exists for instant rollback without a redeploy.
    search_use_embedding_retrieval: bool = True
    search_embedding_top_k: int = 200

    # Secondary Azure OpenAI deployment for A/B comparison (gpt-5.4-mini or
    # whatever is configured). Used only by the /test/compare-vision endpoint
    # — production vision analysis still uses the primary AZURE_OPENAI_*.
    test_azure_openai_api_key: str = ""
    test_azure_openai_endpoint: str = ""
    test_azure_openai_deployment: str = ""
    test_azure_openai_model: str = ""

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "admin"
    postgres_password: str = "admin123"
    postgres_db: str = "real_estate"

    log_level: str = "INFO"
    log_dir: str = "/app/log"  # request/response log location; override for local dev

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
