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
    
    # Research Configuration
    max_iterations: int = Field(default=5, alias="MAX_ITERATIONS")
    max_papers_per_query: int = Field(default=10, alias="MAX_PAPERS_PER_QUERY")
    max_chunks_per_paper: int = Field(default=100, alias="MAX_CHUNKS_PER_PAPER")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    # Coverage Thresholds
    min_claims_per_section: int = Field(default=3, alias="MIN_CLAIMS_PER_SECTION")
    min_papers_for_taxonomy: int = Field(default=5, alias="MIN_PAPERS_FOR_TAXONOMY")
    taxonomy_coverage_threshold: float = Field(default=0.7, alias="TAXONOMY_COVERAGE_THRESHOLD")
    benchmark_coverage_threshold: float = Field(default=0.6, alias="BENCHMARK_COVERAGE_THRESHOLD")
    marginal_gain_threshold: int = Field(default=2, alias="MARGINAL_GAIN_THRESHOLD")
    
    # Output Configuration
    output_language: str = Field(default="en", alias="OUTPUT_LANGUAGE")
    citation_style: Literal["numeric", "author-year"] = Field(default="numeric", alias="CITATION_STYLE")
    
    # Storage paths
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    vector_store_path: str = Field(default="./data/vector_store", alias="VECTOR_STORE_PATH")
    db_path: str = Field(default="./data/research.db", alias="DB_PATH")
    
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
