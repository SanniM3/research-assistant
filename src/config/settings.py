"""Application settings and configuration."""
import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    
    # LLM Configuration
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # Model tiering (cost control): expensive model for generative/critical work,
    # cheap model for high-volume extraction/screening. See agents.base.ROLE_MODELS.
    synthesis_model: str = Field(default="gpt-4o", alias="SYNTHESIS_MODEL")
    extraction_model: str = Field(default="gpt-4o-mini", alias="EXTRACTION_MODEL")
    embedding_batch_size: int = Field(default=128, alias="EMBEDDING_BATCH_SIZE")

    # Cost guardrail: soft cap on estimated LLM spend per run (0 disables).
    max_run_cost_usd: float = Field(default=0.0, alias="MAX_RUN_COST_USD")
    
    # Research Configuration
    max_iterations: int = Field(default=5, alias="MAX_ITERATIONS")
    max_papers_per_query: int = Field(default=10, alias="MAX_PAPERS_PER_QUERY")
    max_chunks_per_paper: int = Field(default=100, alias="MAX_CHUNKS_PER_PAPER")
    # How many chunks to send to the extractor per LLM call (token-bounded batching).
    extraction_chunk_batch: int = Field(default=8, alias="EXTRACTION_CHUNK_BATCH")
    # How many papers to fully extract per research iteration (0 = no cap).
    max_papers_per_extraction: int = Field(default=0, alias="MAX_PAPERS_PER_EXTRACTION")
    chunk_size: int = Field(default=1800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=250, alias="CHUNK_OVERLAP")
    
    # Coverage Thresholds
    min_claims_per_section: int = Field(default=3, alias="MIN_CLAIMS_PER_SECTION")
    min_papers_for_taxonomy: int = Field(default=5, alias="MIN_PAPERS_FOR_TAXONOMY")
    taxonomy_coverage_threshold: float = Field(default=0.7, alias="TAXONOMY_COVERAGE_THRESHOLD")
    benchmark_coverage_threshold: float = Field(default=0.6, alias="BENCHMARK_COVERAGE_THRESHOLD")
    marginal_gain_threshold: int = Field(default=2, alias="MARGINAL_GAIN_THRESHOLD")

    # Retrieval (RAG) Configuration
    retrieval_top_k_claims: int = Field(default=60, alias="RETRIEVAL_TOP_K_CLAIMS")
    retrieval_top_k_chunks: int = Field(default=25, alias="RETRIEVAL_TOP_K_CHUNKS")
    entity_link_similarity: float = Field(default=0.62, alias="ENTITY_LINK_SIMILARITY")
    
    # Output Configuration
    output_language: str = Field(default="en", alias="OUTPUT_LANGUAGE")
    citation_style: Literal["numeric", "author-year"] = Field(default="numeric", alias="CITATION_STYLE")

    # Multilingual: also emit search queries in these languages (comma-separated
    # codes) in addition to the target language and English.
    translate_claims: bool = Field(default=False, alias="TRANSLATE_CLAIMS")
    
    # Storage paths
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    corpus_dir: str = Field(default="./data/corpora", alias="CORPUS_DIR")
    vector_store_path: str = Field(default="./data/vector_store", alias="VECTOR_STORE_PATH")
    db_path: str = Field(default="./data/research.db", alias="DB_PATH")

    # Persistence: when true, the knowledge base persists per-corpus across runs
    # (dynamic KB); when false, each run is ephemeral.
    enable_persistence: bool = Field(default=True, alias="ENABLE_PERSISTENCE")
    corpus_reuse: bool = Field(default=True, alias="CORPUS_REUSE")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/research.log", alias="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
