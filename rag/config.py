"""Configuration management for RAG platform."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """Embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    provider: Literal["openai", "local"] = Field(
        default="openai",
        alias="EMBEDDING_PROVIDER",
    )
    openai_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    local_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )


class LLMConfig(BaseSettings):
    """LLM configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai",
        alias="LLM_PROVIDER",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_MODEL",
    )
    anthropic_model: str = Field(
        default="claude-3-haiku-20240307",
        alias="ANTHROPIC_MODEL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_MODEL",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_URL",
    )
    fallback_enabled: bool = Field(
        default=True,
        alias="LLM_FALLBACK_ENABLED",
    )


class QdrantConfig(BaseSettings):
    """Qdrant configuration."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    url: str = Field(default="http://localhost:6333")
    collection: str = Field(default="rag_chunks")
    api_key: str | None = Field(default=None)


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    rrf_k: int = Field(default=60, alias="RRF_K")
    freshness_half_life_days: float = Field(
        default=30.0,
        alias="FRESHNESS_HALF_LIFE_DAYS",
    )
    freshness_weight: float = Field(
        default=0.2,
        alias="FRESHNESS_WEIGHT",
    )
    initial_limit: int = Field(
        default=100,
        alias="RETRIEVAL_INITIAL_LIMIT",
    )
    rerank_limit: int = Field(
        default=20,
        alias="RETRIEVAL_RERANK_LIMIT",
    )
    final_limit: int = Field(
        default=5,
        alias="RETRIEVAL_FINAL_LIMIT",
    )


class ChunkingSettings(BaseSettings):
    """Chunking configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    default_strategy: str = Field(
        default="recursive",
        alias="DEFAULT_CHUNKING_STRATEGY",
    )


class RerankerSettings(BaseSettings):
    """Reranker configuration."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    model: str = Field(default="BAAI/bge-reranker-base")
    batch_size: int = Field(default=32)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # API Settings
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_upload_size: int = Field(default=52428800, alias="MAX_UPLOAD_SIZE")

    # BM25 Index
    bm25_index_path: str = Field(
        default="data/bm25_index.pkl",
        alias="BM25_INDEX_PATH",
    )

    # Sub-configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None and len(self.anthropic_api_key) > 0

    @property
    def can_use_cloud_embeddings(self) -> bool:
        """Check if cloud embeddings are available."""
        return self.has_openai

    @property
    def can_use_cloud_llm(self) -> bool:
        """Check if any cloud LLM is available."""
        return self.has_openai or self.has_anthropic


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
