from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI (api.openai.com) — single key for vision, geocoding, query parsing, feature filtering, and embeddings.
    openai_api_key: str = ""
    openai_model: str = "gpt-5.1"                # vision + geocoding
    openai_model_for_query: str = "gpt-5.1"      # /search query parsing + feature filter
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    # Secondary model for the /test/compare-vision A/B endpoint (same key).
    openai_test_model: str = ""

    # /search feature retrieval: True → embedding top-K + GPT filter; False → legacy dump-all-features; flag enables instant rollback.
    search_use_embedding_retrieval: bool = True
    search_embedding_top_k: int = 200

    # OpenAI Batch API for photo analysis (50% cost, async ≤24h turnaround). Hybrid:
    # a backlog of ≥ threshold pending properties goes through one batch; smaller
    # uploads stay on the instant sync path. False = current behavior everywhere.
    vision_use_batch: bool = False
    vision_batch_threshold: int = 10     # pending properties needed to trigger a batch
    vision_batch_max_items: int = 150    # max properties per submitted batch
    vision_batch_poll_seconds: int = 60  # min seconds between batch status polls
    # TOTAL estimated-token budget across ALL in-flight batches — size this to your
    # org's Batch Queue Limit (platform.openai.com → Settings → Limits) with ~10%
    # headroom. After a tier raise, bump this env var; no code change needed.
    vision_batch_queue_tokens: int = 800_000
    # Size of ONE batch (estimated tokens). Equal to the queue budget (default) =
    # single-batch waves; smaller = several concurrent batches that refill as each
    # completes (finer progress, smaller failure blast radius).
    vision_batch_max_tokens: int = 800_000
    # Patient mode: while a batch is in flight, batch-eligible rows WAIT for the next
    # wave (everything gets the 50% discount) instead of draining via the full-price
    # sync path. False = latency-optimized (current behavior).
    vision_batch_patient: bool = False

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "admin"
    postgres_password: str = "admin123"
    postgres_db: str = "real_estate"

    log_level: str = "INFO"
    log_dir: str = "/app/log"  # request/response log location; override for local dev

    # env_ignore_empty: a blank line in .env (e.g. `VISION_USE_BATCH=` copied from
    # .env.example) means "use the default" instead of crashing bool/int parsing.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "env_ignore_empty": True}


settings = Settings()
