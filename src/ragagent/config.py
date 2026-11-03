"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

PROVIDERS = ("fake", "openai", "azure")


@dataclass(frozen=True)
class Settings:
    provider: str = "fake"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    azure_chat_deployment: str | None = None
    azure_embedding_deployment: str | None = None
    azure_api_version: str = "2024-10-21"
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 150


def get_settings() -> Settings:
    """Build settings from the environment. Called at runtime, so tests can patch env vars."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "fake").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
    return Settings(
        provider=provider,
        chat_model=os.getenv("CHAT_MODEL", Settings.chat_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        azure_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", Settings.azure_api_version),
        top_k=int(os.getenv("TOP_K", Settings.top_k)),
        chunk_size=int(os.getenv("CHUNK_SIZE", Settings.chunk_size)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", Settings.chunk_overlap)),
    )
