"""
FastAPI Routes for RAG Engine.

=============================================================================
THEORY: RAG API Design
=============================================================================

A RAG system typically needs these endpoints:

1. Query Endpoints:
   - POST /query: Ask questions, get answers with sources
   - POST /search: Direct vector search (no LLM)

2. Ingestion Endpoints:
   - POST /ingest/text: Ingest raw text
   - POST /ingest/file: Ingest a file
   - POST /ingest/directory: Ingest all files in directory

3. Management Endpoints:
   - GET /knowledge-base/stats: View indexed documents
   - DELETE /knowledge-base/source/{source}: Remove a source
   - DELETE /knowledge-base: Clear all data

=============================================================================
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import json

from ..config import Settings, get_settings
from ..core.chunker import DocumentLoader, TextChunker
from ..core.embeddings import BedrockEmbeddings
from ..core.retriever import Retriever
from ..core.rag_pipeline import RAGPipeline, IngestionPipeline
from ..storage.vector_store import ChromaVectorStore
from .models import (
    QueryRequest,
    QueryResponse,
    SourceInfo,
    IngestTextRequest,
    IngestResponse,
    BulkIngestResponse,
    KnowledgeBaseStats,
    SearchRequest,
    SearchResult,
    SearchResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Dependency Injection
# =============================================================================

def get_embeddings(settings: Settings = Depends(get_settings)) -> BedrockEmbeddings:
    """Get embedding service."""
    return BedrockEmbeddings(
        region_name=settings.aws_region,
        model_id=settings.embedding_model,
        profile_name=settings.aws_profile or None,
        dimensions=settings.embedding_dimension,
    )


def get_vector_store(settings: Settings = Depends(get_settings)) -> ChromaVectorStore:
    """Get vector store."""
    return ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
    )


def get_chunker(settings: Settings = Depends(get_settings)) -> TextChunker:
    """Get text chunker."""
    return TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_retriever(
    embeddings: BedrockEmbeddings = Depends(get_embeddings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> Retriever:
    """Get retriever."""
    return Retriever(
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=settings.top_k_results,
        min_similarity=settings.min_similarity,
    )


def get_rag_pipeline(
    retriever: Retriever = Depends(get_retriever),
    settings: Settings = Depends(get_settings),
) -> RAGPipeline:
    """Get RAG pipeline."""
    return RAGPipeline(
        retriever=retriever,
        gateway_url=settings.ai_gateway_url,
    )


def get_ingestion_pipeline(
    embeddings: BedrockEmbeddings = Depends(get_embeddings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    chunker: TextChunker = Depends(get_chunker),
) -> IngestionPipeline:
    """Get ingestion pipeline."""
    return IngestionPipeline(
        embeddings=embeddings,
        vector_store=vector_store,
        chunker=chunker,
    )


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-engine"}


# =============================================================================
# Query Endpoints
# =============================================================================

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Query the knowledge base.
    
    This is the main RAG endpoint:
    1. Retrieves relevant context from the knowledge base
    2. Sends query + context to the LLM (via Week 1 Gateway)
    3. Returns answer with sources
    """
    if request.stream:
        return await query_stream(request, rag_pipeline)
    
    start_time = time.time()
    
    try:
        response = await rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            model=request.model,
            temperature=request.temperature,
            filter_source=request.filter_source,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=response.answer,
            sources=[
                SourceInfo(
                    source=s.source,
                    similarity=s.similarity,
                    chunk_index=s.metadata.get("chunk_index"),
                )
                for s in response.sources
            ],
            tokens_used=response.tokens_used,
            cost=response.cost,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def query_stream(
    request: QueryRequest,
    rag_pipeline: RAGPipeline,
):
    """Stream RAG response."""
    
    async def event_generator():
        try:
            async for chunk in rag_pipeline.query_stream(
                question=request.question,
                top_k=request.top_k,
                model=request.model,
                temperature=request.temperature,
            ):
                yield {
                    "event": chunk["event"],
                    "data": json.dumps(chunk["data"]) if isinstance(chunk["data"], (dict, list)) else chunk["data"],
                }
        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
    
    return EventSourceResponse(event_generator())


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
):
    """
    Direct vector search without LLM generation.
    
    Useful for:
    - Testing retrieval quality
    - Finding similar documents
    - Debugging
    """
    start_time = time.time()
    
    try:
        results = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            filter_source=request.filter_source,
        )
        
        return SearchResponse(
            results=[
                SearchResult(
                    content=r.content,
                    source=r.source,
                    similarity=r.similarity,
                    metadata=r.metadata,
                )
                for r in results
            ],
            query=request.query,
            total_results=len(results),
        )
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Ingestion Endpoints
# =============================================================================

@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    request: IngestTextRequest,
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """
    Ingest raw text into the knowledge base.
    
    Example use:
    - Pasting content directly
    - API integrations
    - Testing
    """
    try:
        result = await ingestion_pipeline.ingest_text(
            text=request.text,
            source=request.source,
            metadata=request.metadata,
        )
        
        return IngestResponse(**result)
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file_path: str,
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """
    Ingest a file from the local filesystem.
    
    Supported formats: .txt, .md, .pdf, .docx, .html
    """
    try:
        loader = DocumentLoader()
        document = loader.load_file(file_path)
        result = await ingestion_pipeline.ingest_document(document)
        
        return IngestResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/directory", response_model=BulkIngestResponse)
async def ingest_directory(
    dir_path: str,
    recursive: bool = True,
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """
    Ingest all supported files from a directory.
    """
    try:
        loader = DocumentLoader()
        documents = loader.load_directory(dir_path, recursive=recursive)
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No supported files found in: {dir_path}",
            )
        
        result = await ingestion_pipeline.ingest_documents(documents)
        
        return BulkIngestResponse(
            total_documents=result["total_documents"],
            total_chunks=result["total_chunks"],
            results=[IngestResponse(**r) for r in result["results"]],
        )
    except NotADirectoryError:
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    except Exception as e:
        logger.exception(f"Bulk ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """
    Upload and ingest a file.
    
    Accepts file upload via multipart form.
    """
    import tempfile
    import os
    
    # Check file extension
    filename = file.filename or "uploaded_file"
    extension = Path(filename).suffix.lower()
    
    if extension not in DocumentLoader.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported: {DocumentLoader.SUPPORTED_EXTENSIONS}",
        )
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load and ingest
        loader = DocumentLoader()
        document = loader.load_file(tmp_path)
        document.source = filename  # Use original filename
        
        result = await ingestion_pipeline.ingest_document(document)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return IngestResponse(**result)
    except Exception as e:
        logger.exception(f"Upload ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Knowledge Base Management
# =============================================================================

@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStats)
async def get_stats(
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Get statistics about the knowledge base."""
    stats = vector_store.get_stats()
    
    return KnowledgeBaseStats(
        collection_name=stats["collection_name"],
        total_chunks=stats["total_chunks"],
        total_sources=len(stats["sources"]),
        sources=stats["sources"],
    )


@router.delete("/knowledge-base/source/{source:path}")
async def delete_source(
    source: str,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Delete all chunks from a specific source."""
    deleted = vector_store.delete_by_source(source)
    
    return {
        "source": source,
        "chunks_deleted": deleted,
        "status": "success" if deleted > 0 else "not_found",
    }


@router.delete("/knowledge-base")
async def clear_knowledge_base(
    confirm: bool = False,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """
    Clear all data from the knowledge base.
    
    Requires confirm=true to prevent accidental deletion.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm deletion of all data",
        )
    
    vector_store.clear()
    
    return {"status": "cleared", "message": "All data has been deleted"}
