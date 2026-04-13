"""Centralized configuration using Pydantic Settings.

Reads from environment variables and .env file.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Providers ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Search APIs ---
    tavily_api_key: str = ""

    # --- Vector Store ---
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "atlas_documents"

    # --- Embedding Model ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- LLM Config ---
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # --- Retrieval Config ---
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 5

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"

    # --- Paths ---
    data_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data")
    eval_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "eval")


# Singleton instance
settings = Settings()
