"""
Retriever - The core of RAG.

=============================================================================
THEORY: Retrieval in RAG Systems
=============================================================================

The retriever is responsible for finding relevant context for a query.

RAG Pipeline:
-------------
Query → Embed Query → Search Vector Store → Retrieve Top-K → Build Context → LLM

Why Retrieval Matters:
----------------------
1. Garbage In, Garbage Out: Wrong context = Wrong answers
2. Cost: More context = More tokens = Higher cost
3. Quality: Right context enables accurate, grounded responses

Retrieval Strategies:
---------------------

1. Dense Retrieval (what we use):
   - Embed query and documents
   - Find nearest neighbors by vector similarity
   - Good for semantic matching
   
2. Sparse Retrieval (BM25):
   - Traditional keyword matching with TF-IDF
   - Good for exact term matching
   - Fast and interpretable
   
3. Hybrid Retrieval:
   - Combine dense + sparse
   - Best of both worlds
   - More complex to implement

4. Re-ranking:
   - Retrieve more candidates (top-50)
   - Re-rank with a cross-encoder model
   - Return top-k after re-ranking
   - Better quality, more compute

Retrieval Metrics:
------------------
- Recall@K: % of relevant docs in top-K
- Precision@K: % of top-K that are relevant
- MRR: Mean Reciprocal Rank (where is first relevant result?)
- NDCG: Normalized Discounted Cumulative Gain

=============================================================================
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .embeddings import BedrockEmbeddings
from ..storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    A single retrieval result.
    
    Contains the chunk content, its source, and relevance score.
    """
    content: str
    source: str
    similarity: float
    metadata: dict
    
    def __str__(self) -> str:
        return f"[{self.similarity:.2f}] {self.source}: {self.content[:100]}..."


class Retriever:
    """
    Retrieves relevant context for queries.
    
    THEORY: Retriever Design
    ------------------------
    
    The retriever connects:
    1. Query processing (embedding)
    2. Vector search (similarity)
    3. Result formatting (for LLM)
    
    Configuration considerations:
    - top_k: More results = more context but higher cost
    - min_similarity: Filter out irrelevant matches
    - filter_metadata: Scope search to specific sources
    """
    
    def __init__(
        self,
        embeddings: BedrockEmbeddings,
        vector_store: ChromaVectorStore,
        top_k: int = 5,
        min_similarity: float = 0.7,
    ):
        """
        Initialize retriever.
        
        Args:
            embeddings: Embedding service for query encoding
            vector_store: Vector store for similarity search
            top_k: Number of results to retrieve
            min_similarity: Minimum similarity threshold
        """
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._top_k = top_k
        self._min_similarity = min_similarity
        
        logger.info(f"Retriever initialized: top_k={top_k}, min_similarity={min_similarity}")
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filter_source: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        THEORY: Query Processing
        ------------------------
        
        Steps:
        1. Embed the query (same model as documents)
        2. Search vector store for similar embeddings
        3. Filter by similarity threshold
        4. Return formatted results
        
        Args:
            query: User's question
            top_k: Override default top_k
            min_similarity: Override default threshold
            filter_source: Only search specific source
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self._top_k
        min_similarity = min_similarity or self._min_similarity
        
        logger.info(f"Retrieving for query: '{query[:50]}...'")
        
        # Step 1: Embed the query
        query_embedding = await self._embeddings.embed_query(query)
        
        # Step 2: Search vector store
        filter_metadata = {"source": filter_source} if filter_source else None
        
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            filter_metadata=filter_metadata,
        )
        
        # Step 3: Format results
        retrieval_results = [
            RetrievalResult(
                content=r["content"],
                source=r["metadata"].get("source", "unknown"),
                similarity=r["similarity"],
                metadata=r["metadata"],
            )
            for r in results
        ]
        
        logger.info(f"Retrieved {len(retrieval_results)} relevant chunks")
        return retrieval_results
    
    def format_context(
        self,
        results: list[RetrievalResult],
        max_tokens: int = 4000,
    ) -> str:
        """
        Format retrieval results as context for LLM.
        
        THEORY: Context Formatting
        --------------------------
        
        How you present retrieved context to the LLM matters:
        
        Bad (no structure):
        "Python is a programming language. Machine learning uses Python. 
         Variables store data. Functions are reusable code blocks."
        
        Good (structured with sources):
        "[Source: python_tutorial.pdf]
         Python is a programming language known for its simplicity.
         
         [Source: ml_basics.md]
         Machine learning commonly uses Python due to its rich ecosystem."
        
        Best practices:
        - Include source attribution
        - Group by source
        - Indicate relevance scores
        - Limit total length
        
        Args:
            results: List of retrieval results
            max_tokens: Approximate max tokens for context
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        # Estimate chars per token (rough approximation)
        max_chars = max_tokens * 4
        
        context_parts = []
        current_chars = 0
        
        for i, result in enumerate(results, 1):
            # Format each chunk with metadata
            chunk_text = (
                f"[Context {i}]\n"
                f"Source: {result.source}\n"
                f"Relevance: {result.similarity:.0%}\n"
                f"Content:\n{result.content}\n"
            )
            
            # Check if we're exceeding limit
            if current_chars + len(chunk_text) > max_chars:
                # Truncate this chunk if needed
                remaining = max_chars - current_chars - 100  # Buffer for formatting
                if remaining > 200:  # Only include if meaningful
                    truncated = result.content[:remaining] + "..."
                    chunk_text = (
                        f"[Context {i}]\n"
                        f"Source: {result.source}\n"
                        f"Content:\n{truncated}\n"
                    )
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            current_chars += len(chunk_text)
        
        return "\n---\n".join(context_parts)
    
    async def retrieve_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_context_tokens: int = 4000,
    ) -> tuple[str, list[RetrievalResult]]:
        """
        Retrieve and format context in one call.
        
        Convenience method that combines retrieve() and format_context().
        
        Args:
            query: User's question
            top_k: Number of results
            max_context_tokens: Max tokens for formatted context
            
        Returns:
            Tuple of (formatted_context, raw_results)
        """
        results = await self.retrieve(query, top_k=top_k)
        context = self.format_context(results, max_tokens=max_context_tokens)
        return context, results
