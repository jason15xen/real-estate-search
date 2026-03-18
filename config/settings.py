from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure-hosted Anthropic endpoints (Opus and Haiku use different Azure services)
    azure_anthropic_endpoint: str = "https://v-cke-m8mjd1dx-eastus2.services.ai.azure.com/anthropic/"
    azure_anthropic_endpoint_fast: str = "https://v-cke-m8mjd1dx-eastus2.openai.azure.com/anthropic/"
    azure_anthropic_api_key: str = ""

    # Models (Azure-hosted)
    anthropic_model: str = "claude-opus-4-6"
    anthropic_model_fast: str = "claude-haiku-4-5"

    chroma_persist_dir: str = "./chroma_db"
    log_level: str = "INFO"

    # Search tuning
    vector_search_top_k: int = 50
    vector_similarity_threshold: float = 0.3

    # Validation
    max_candidates_for_validation: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
