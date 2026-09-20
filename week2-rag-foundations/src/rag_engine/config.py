"""
Configuration Management for RAG Engine.

=============================================================================
THEORY: RAG System Configuration
=============================================================================

Key configuration areas for RAG systems:

1. Chunking Parameters:
   - chunk_size: How big each text piece should be
   - chunk_overlap: How much overlap between chunks (prevents losing context)

2. Embedding Settings:
   - Which model to use (Titan, OpenAI ada-002, etc.)
   - Dimension of embeddings (affects storage and search)

3. Retrieval Parameters:
   - top_k: How many chunks to retrieve
   - min_similarity: Threshold for relevance

4. Vector Store:
   - Where to persist data
   - Collection naming
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RAG Engine configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # =========================================================================
    # AWS Configuration
    # =========================================================================
    aws_profile: str = "ai-learning"
    aws_region: str = "us-east-1"
    
    # =========================================================================
    # AI Gateway Connection (Week 1)
    # =========================================================================
    ai_gateway_url: str = "http://localhost:8000"
    
    # =========================================================================
    # Vector Database
    # =========================================================================
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "knowledge_base"
    
    # =========================================================================
    # Document Processing
    # =========================================================================
    # Chunk size in characters
    # Rule of thumb: 1000 chars ≈ 250 tokens
    chunk_size: int = 1000
    
    # Overlap prevents losing context at chunk boundaries
    # 20% overlap is a good starting point
    chunk_overlap: int = 200
    
    # =========================================================================
    # Retrieval Settings
    # =========================================================================
    # Number of chunks to retrieve
    top_k_results: int = 5
    
    # Minimum similarity score (0-1, higher = more strict)
    min_similarity: float = 0.7
    
    # =========================================================================
    # Embedding Model
    # =========================================================================
    # Amazon Titan Embeddings V2
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    
    # Embedding dimension (Titan V2 = 1024)
    embedding_dimension: int = 1024
    
    # =========================================================================
    # Server Configuration
    # =========================================================================
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
