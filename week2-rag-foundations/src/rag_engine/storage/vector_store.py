"""
Vector Store using ChromaDB.

=============================================================================
THEORY: What is a Vector Database?
=============================================================================

A vector database stores and retrieves data based on vector similarity,
not exact matches or keyword search.

Traditional Database:
    Query: "SELECT * FROM docs WHERE title = 'Python Tutorial'"
    Match: Exact string match only

Vector Database:
    Query: embed("How do I learn programming?")
    Match: Returns documents semantically similar (Python tutorials, coding guides, etc.)

Why Vector Databases for RAG:
-----------------------------
1. Semantic Search: Find conceptually similar content
2. Scale: Efficiently search millions of vectors
3. Speed: Approximate nearest neighbor (ANN) algorithms
4. Metadata Filtering: Combine vector search with filters

Popular Vector Databases:
-------------------------
| Name | Type | Best For |
|------|------|----------|
| ChromaDB | Embedded | Development, small-medium scale |
| Pinecone | Cloud | Production, managed service |
| Weaviate | Self-hosted | Full-featured, GraphQL API |
| Qdrant | Self-hosted | Rust performance, filters |
| pgvector | PostgreSQL | Existing Postgres users |
| FAISS | Library | Maximum performance, research |

ChromaDB:
---------
- Embedded (runs in-process) or client-server
- Supports persistence to disk
- Built-in embedding functions
- Simple Python API
- Great for learning and prototypes

=============================================================================
"""

import logging
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..core.chunker import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Vector store implementation using ChromaDB.
    
    THEORY: ChromaDB Architecture
    -----------------------------
    ChromaDB stores:
    1. Embeddings: The vector representations
    2. Documents: Original text content
    3. Metadata: Key-value pairs for filtering
    4. IDs: Unique identifiers
    
    Index types (for similarity search):
    - HNSW (default): Fast approximate search
    - Flat: Exact search (slower but precise)
    
    Distance metrics:
    - cosine (default): Angle-based, good for normalized vectors
    - l2: Euclidean distance
    - ip: Inner product
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "knowledge_base",
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Where to store the database
            collection_name: Name of the collection
        """
        self._persist_dir = Path(persist_directory)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )
        
        self._collection_name = collection_name
        self._collection = None
        
        logger.info(f"ChromaDB initialized at {self._persist_dir}")
    
    def get_or_create_collection(
        self,
        embedding_dimension: int = 1024,
    ) -> chromadb.Collection:
        """
        Get existing collection or create new one.
        
        THEORY: Collection Configuration
        ---------------------------------
        Collections are like tables in a traditional database.
        
        Metadata options:
        - hnsw:space: Distance metric (cosine, l2, ip)
        - hnsw:M: Connections per node (higher = more accurate but slower)
        - hnsw:construction_ef: Build-time accuracy (higher = better index)
        - hnsw:search_ef: Search-time accuracy (higher = more accurate)
        """
        if self._collection is not None:
            return self._collection
        
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity
                "description": "RAG knowledge base",
            },
        )
        
        logger.info(
            f"Collection '{self._collection_name}' ready: "
            f"{self._collection.count()} documents"
        )
        return self._collection
    
    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add chunks with their embeddings to the store.
        
        Args:
            chunks: List of text chunks
            embeddings: Corresponding embeddings (same order)
            
        Returns:
            Number of chunks added
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            return 0
        
        collection = self.get_or_create_collection()
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                **{k: str(v) for k, v in chunk.metadata.items()},  # Convert all to string
            }
            for chunk in chunks
        ]
        
        # Add to collection (upsert to handle duplicates)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(chunks)} chunks to collection")
        return len(chunks)
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar chunks.
        
        THEORY: Approximate Nearest Neighbor (ANN) Search
        -------------------------------------------------
        Finding exact nearest neighbors in high-dimensional space is slow.
        ANN algorithms trade accuracy for speed:
        
        - HNSW: Hierarchical Navigable Small World graphs
        - IVF: Inverted File Index (clusters)
        - PQ: Product Quantization (compression)
        
        ChromaDB uses HNSW by default, which is very fast and accurate.
        
        Args:
            query_embedding: Vector to search for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            filter_metadata: Filter by metadata (e.g., {"source": "doc.pdf"})
            
        Returns:
            List of results with content, metadata, and similarity score
        """
        collection = self.get_or_create_collection()
        
        # Build where clause for metadata filtering
        where = filter_metadata if filter_metadata else None
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Process results
        processed_results = []
        
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # ChromaDB returns distance, convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                # Filter by minimum similarity
                if similarity < min_similarity:
                    continue
                
                processed_results.append({
                    "id": doc_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": round(similarity, 4),
                })
        
        logger.debug(f"Search returned {len(processed_results)} results")
        return processed_results
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source document.
        
        Args:
            source: Source identifier (e.g., file path)
            
        Returns:
            Number of chunks deleted
        """
        collection = self.get_or_create_collection()
        
        # Get IDs to delete
        results = collection.get(
            where={"source": source},
            include=[],
        )
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from source: {source}")
            return len(results['ids'])
        
        return 0
    
    def get_all_sources(self) -> list[str]:
        """Get list of all unique source documents."""
        collection = self.get_or_create_collection()
        
        # Get all metadatas
        results = collection.get(include=["metadatas"])
        
        sources = set()
        for metadata in results.get('metadatas', []):
            if metadata and 'source' in metadata:
                sources.add(metadata['source'])
        
        return sorted(sources)
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        collection = self.get_or_create_collection()
        
        return {
            "collection_name": self._collection_name,
            "total_chunks": collection.count(),
            "persist_directory": str(self._persist_dir),
            "sources": self.get_all_sources(),
        }
    
    def clear(self) -> None:
        """Delete all data in the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = None
        logger.warning(f"Cleared collection: {self._collection_name}")
