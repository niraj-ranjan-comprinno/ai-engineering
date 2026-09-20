"""
RAG Engine - Main Application Entry Point.

=============================================================================
THEORY: RAG System Architecture
=============================================================================

This application provides a complete RAG (Retrieval-Augmented Generation) system:

┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG ENGINE (Port 8001)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        FastAPI Application                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │  /query  │  │ /search  │  │ /ingest  │  │ /knowledge-base  │ │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │  │
│  └───────┼─────────────┼─────────────┼─────────────────┼───────────┘  │
│          │             │             │                 │              │
│          ▼             ▼             ▼                 ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         Core Components                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │  │
│  │  │ RAGPipeline│  │ Retriever  │  │ Embeddings │  │  Chunker   │ │  │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────────────┘ │  │
│  └────────┼───────────────┼───────────────┼─────────────────────────┘  │
│           │               │               │                            │
│           ▼               ▼               ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                          Storage Layer                            │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │                    ChromaDB Vector Store                     │ │  │
│  │  │                    (Persistent Storage)                      │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │   AWS Bedrock    │  │   AI Gateway     │  │   Documents      │
         │   (Embeddings)   │  │   (Week 1)       │  │   (Your Data)    │
         │   Titan V2       │  │   Port 8000      │  │   PDF/MD/TXT     │
         └──────────────────┘  └──────────────────┘  └──────────────────┘

=============================================================================
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
    - Log configuration
    - Verify dependencies
    
    Shutdown:
    - Cleanup resources
    """
    settings = get_settings()
    
    # Startup
    logger.info("=" * 60)
    logger.info("RAG Engine starting up...")
    logger.info(f"  ChromaDB: {settings.chroma_persist_dir}")
    logger.info(f"  Collection: {settings.collection_name}")
    logger.info(f"  AI Gateway: {settings.ai_gateway_url}")
    logger.info(f"  Embedding Model: {settings.embedding_model}")
    logger.info(f"  Chunk Size: {settings.chunk_size}, Overlap: {settings.chunk_overlap}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("RAG Engine shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RAG Engine",
    description="Week 2: RAG Foundations - Knowledge Retrieval System",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/v1", tags=["rag"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "RAG Engine",
        "version": "0.1.0",
        "week": "Week 2: RAG Foundations",
        "endpoints": {
            "query": "/v1/query",
            "search": "/v1/search",
            "ingest_text": "/v1/ingest/text",
            "ingest_file": "/v1/ingest/file",
            "ingest_directory": "/v1/ingest/directory",
            "ingest_upload": "/v1/ingest/upload",
            "stats": "/v1/knowledge-base/stats",
            "health": "/v1/health",
        },
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "rag_engine.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
