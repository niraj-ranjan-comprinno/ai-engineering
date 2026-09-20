"""
Pydantic Models for RAG API.

Request and response schemas for the RAG endpoints.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Query Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request for RAG query."""
    question: str = Field(
        description="The question to answer",
        min_length=1,
        max_length=5000,
    )
    top_k: int = Field(
        default=5,
        description="Number of context chunks to retrieve",
        ge=1,
        le=20,
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model to use (defaults to server default)",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Generation temperature",
        ge=0.0,
        le=2.0,
    )
    filter_source: Optional[str] = Field(
        default=None,
        description="Only search specific source document",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )


class SourceInfo(BaseModel):
    """Information about a source document."""
    source: str
    similarity: float
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str = Field(description="Generated answer")
    sources: list[SourceInfo] = Field(description="Sources used for answer")
    tokens_used: dict = Field(description="Token usage statistics")
    cost: Optional[dict] = Field(default=None, description="Cost breakdown")
    latency_ms: float = Field(description="Query latency in milliseconds")


# =============================================================================
# Ingestion Models
# =============================================================================

class IngestTextRequest(BaseModel):
    """Request to ingest raw text."""
    text: str = Field(
        description="Text content to ingest",
        min_length=1,
    )
    source: str = Field(
        default="manual_input",
        description="Source identifier",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata",
    )


class IngestRequest(BaseModel):
    """Request to ingest a file."""
    file_path: str = Field(
        description="Path to file to ingest",
    )


class IngestResponse(BaseModel):
    """Response from ingestion."""
    source: str = Field(description="Source identifier")
    chunks: int = Field(description="Number of chunks created")
    status: str = Field(description="Ingestion status")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BulkIngestResponse(BaseModel):
    """Response from bulk ingestion."""
    total_documents: int
    total_chunks: int
    results: list[IngestResponse]


# =============================================================================
# Knowledge Base Models
# =============================================================================

class KnowledgeBaseStats(BaseModel):
    """Statistics about the knowledge base."""
    collection_name: str
    total_chunks: int
    total_sources: int
    sources: list[str]


class SearchRequest(BaseModel):
    """Request for direct vector search (no LLM)."""
    query: str = Field(
        description="Search query",
        min_length=1,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )
    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )
    filter_source: Optional[str] = None


class SearchResult(BaseModel):
    """A single search result."""
    content: str
    source: str
    similarity: float
    metadata: dict


class SearchResponse(BaseModel):
    """Response from vector search."""
    results: list[SearchResult]
    query: str
    total_results: int
