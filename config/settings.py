from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI (api.openai.com) — a SINGLE key for the whole app: vision analysis,
    # geocoding, /search query parsing, feature filtering, and embeddings.
    openai_api_key: str = ""
    openai_model: str = "gpt-5.1"                # vision + geocoding
    openai_model_for_query: str = "gpt-5.1"      # /search query parsing + feature filter
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    # Secondary model for the /test/compare-vision A/B endpoint (same key).
    openai_test_model: str = ""

    # /search feature-retrieval strategy:
    #   True  → embedding retrieval (top-K candidates) + GPT relevance filter
    #   False → legacy: dump every DB feature into the GPT query prompt
    # Flag exists for instant rollback without a redeploy.
    search_use_embedding_retrieval: bool = True
    search_embedding_top_k: int = 200

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
