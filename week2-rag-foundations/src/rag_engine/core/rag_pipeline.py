"""
RAG Pipeline - Connecting Retrieval to Generation.

=============================================================================
THEORY: The Complete RAG Pipeline
=============================================================================

RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation:

┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Query                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────────────┐                                                   │
│   │ 1. EMBED QUERY  │  Convert query to vector                         │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ 2. RETRIEVE     │  Find similar chunks in vector store             │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ 3. BUILD PROMPT │  Combine query + retrieved context               │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ 4. GENERATE     │  Send to LLM (via Week 1 Gateway)               │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   Response with Citations                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Why RAG Over Fine-Tuning?
-------------------------
| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge Update | Add docs, instant | Retrain model |
| Cost | Cheap (retrieval) | Expensive (training) |
| Hallucination | Grounded in docs | Can still hallucinate |
| Transparency | Can cite sources | Black box |
| Domain Scope | Limited to indexed docs | Full model capability |

When to Use RAG:
- You have specific documents to query
- Knowledge changes frequently
- You need citations/sources
- You want to control what the model knows

When to Fine-Tune:
- You need to change model behavior/style
- Specific task format (JSON output, etc.)
- Performance optimization

=============================================================================
"""

import logging
from dataclasses import dataclass
from typing import Optional, AsyncIterator

import httpx

from .retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the context below.
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question."
3. Always cite your sources by mentioning which context you used.
4. Be concise but thorough.
5. If multiple contexts provide relevant information, synthesize them.

CONTEXT:
{context}
"""

RAG_USER_PROMPT = """Question: {question}

Please provide a well-structured answer based on the context above."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RAGResponse:
    """
    Complete RAG response with answer and sources.
    """
    answer: str
    sources: list[RetrievalResult]
    context_used: str
    tokens_used: dict
    cost: Optional[dict] = None
    
    def __str__(self) -> str:
        source_list = "\n".join(f"  - {s.source}" for s in self.sources)
        return f"Answer: {self.answer}\n\nSources:\n{source_list}"


# =============================================================================
# RAG Pipeline
# =============================================================================

class RAGPipeline:
    """
    Complete RAG pipeline connecting retrieval to LLM generation.
    
    THEORY: Pipeline Design
    -----------------------
    
    The pipeline orchestrates:
    1. Retriever: Gets relevant context
    2. Prompt Builder: Formats the prompt
    3. LLM Client: Generates the answer
    
    Using Week 1's AI Gateway for LLM calls gives us:
    - Provider abstraction (can switch models)
    - Cost tracking
    - Rate limiting
    - Fallback
    """
    
    def __init__(
        self,
        retriever: Retriever,
        gateway_url: str = "http://localhost:8000",
        default_model: str = "claude-sonnet-4-6",
        temperature: float = 0.3,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Configured retriever instance
            gateway_url: URL of Week 1 AI Gateway
            default_model: Default model to use
            temperature: Generation temperature (low for factual answers)
        """
        self._retriever = retriever
        self._gateway_url = gateway_url.rstrip("/")
        self._default_model = default_model
        self._temperature = temperature
        
        # HTTP client for gateway calls
        self._http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info(f"RAGPipeline initialized with gateway: {gateway_url}")
    
    async def close(self):
        """Close HTTP client."""
        await self._http_client.aclose()
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        filter_source: Optional[str] = None,
    ) -> RAGResponse:
        """
        Run the complete RAG pipeline.
        
        THEORY: RAG Query Flow
        ----------------------
        
        1. Retrieve relevant chunks
        2. Build augmented prompt
        3. Call LLM via Gateway
        4. Return response with sources
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            model: LLM model to use
            temperature: Generation temperature
            filter_source: Only search specific source
            
        Returns:
            RAGResponse with answer and sources
        """
        model = model or self._default_model
        temperature = temperature if temperature is not None else self._temperature
        
        logger.info(f"RAG query: '{question[:50]}...'")
        
        # Step 1: Retrieve relevant context
        results = await self._retriever.retrieve(
            query=question,
            top_k=top_k,
            filter_source=filter_source,
        )
        
        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the knowledge base to answer this question.",
                sources=[],
                context_used="",
                tokens_used={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
        
        # Step 2: Format context
        context = self._retriever.format_context(results)
        
        # Step 3: Build prompt
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        user_prompt = RAG_USER_PROMPT.format(question=question)
        
        # Step 4: Call LLM via Gateway
        response = await self._call_gateway(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )
        
        # Build response
        return RAGResponse(
            answer=response["content"],
            sources=results,
            context_used=context,
            tokens_used=response.get("usage", {}),
            cost=response.get("cost"),
        )
    
    async def query_stream(
        self,
        question: str,
        top_k: int = 5,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[dict]:
        """
        Stream RAG response.
        
        THEORY: Streaming RAG
        ---------------------
        
        For better UX, stream the LLM response while it generates:
        1. First yield: Retrieved sources (immediate feedback)
        2. Then yield: Answer tokens as they arrive
        3. Finally yield: Complete metadata
        
        Yields:
            Dict with event type and data
        """
        model = model or self._default_model
        temperature = temperature if temperature is not None else self._temperature
        
        # Step 1: Retrieve
        results = await self._retriever.retrieve(question, top_k=top_k)
        context = self._retriever.format_context(results)
        
        # Yield sources immediately
        yield {
            "event": "sources",
            "data": {
                "count": len(results),
                "sources": [{"source": r.source, "similarity": r.similarity} for r in results],
            },
        }
        
        if not results:
            yield {
                "event": "content",
                "data": "I couldn't find any relevant information to answer this question.",
            }
            yield {"event": "done", "data": {}}
            return
        
        # Step 2: Build prompt
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        user_prompt = RAG_USER_PROMPT.format(question=question)
        
        # Step 3: Stream from Gateway
        async for chunk in self._stream_gateway(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        ):
            yield chunk
    
    async def _call_gateway(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
    ) -> dict:
        """
        Call Week 1 AI Gateway (non-streaming).
        
        THEORY: Gateway Integration
        ---------------------------
        
        We use the Gateway instead of calling Bedrock directly because:
        1. Centralized API key management
        2. Cost tracking
        3. Provider abstraction
        4. Logging and metrics
        
        The Gateway exposes a standard OpenAI-compatible API.
        """
        url = f"{self._gateway_url}/v1/chat/completions"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": model,
            "temperature": temperature,
            "stream": False,
        }
        
        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def _stream_gateway(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
    ) -> AsyncIterator[dict]:
        """
        Stream from Week 1 AI Gateway.
        
        Parses SSE (Server-Sent Events) from the gateway.
        """
        url = f"{self._gateway_url}/v1/chat/completions"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": model,
            "temperature": temperature,
            "stream": True,
        }
        
        async with self._http_client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                
                data = line[5:].strip()  # Remove "data:" prefix
                
                if data == "[DONE]":
                    yield {"event": "done", "data": {}}
                    break
                
                try:
                    import json
                    parsed = json.loads(data)
                    
                    if "delta" in parsed:
                        yield {"event": "content", "data": parsed["delta"]}
                    elif "usage" in parsed:
                        yield {
                            "event": "complete",
                            "data": {
                                "usage": parsed.get("usage"),
                                "cost": parsed.get("cost"),
                            },
                        }
                except Exception as e:
                    logger.debug(f"Failed to parse SSE line: {e}")


# =============================================================================
# Ingestion Pipeline
# =============================================================================

class IngestionPipeline:
    """
    Pipeline for ingesting documents into the RAG system.
    
    THEORY: Document Ingestion
    --------------------------
    
    Ingestion flow:
    1. Load document(s) from file/directory
    2. Split into chunks
    3. Generate embeddings
    4. Store in vector database
    
    Considerations:
    - Batch processing for efficiency
    - Progress tracking for large documents
    - Error handling per document
    - Duplicate detection
    """
    
    def __init__(
        self,
        embeddings,
        vector_store,
        chunker,
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            embeddings: Embedding service
            vector_store: Vector store
            chunker: Text chunker
        """
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._chunker = chunker
        
        logger.info("IngestionPipeline initialized")
    
    async def ingest_document(self, document) -> dict:
        """
        Ingest a single document.
        
        Args:
            document: Document object to ingest
            
        Returns:
            Dict with ingestion stats
        """
        from .chunker import Document
        
        logger.info(f"Ingesting document: {document.source}")
        
        # Step 1: Chunk the document
        chunks = self._chunker.chunk_document(document)
        
        if not chunks:
            return {
                "source": document.source,
                "chunks": 0,
                "status": "empty",
            }
        
        # Step 2: Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self._embeddings.embed_texts(chunk_texts)
        
        # Step 3: Store in vector database
        added = self._vector_store.add_chunks(chunks, embeddings)
        
        return {
            "source": document.source,
            "chunks": added,
            "status": "success",
        }
    
    async def ingest_documents(self, documents: list) -> dict:
        """
        Ingest multiple documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Summary of ingestion
        """
        results = []
        total_chunks = 0
        
        for doc in documents:
            try:
                result = await self.ingest_document(doc)
                results.append(result)
                total_chunks += result["chunks"]
            except Exception as e:
                logger.error(f"Failed to ingest {doc.source}: {e}")
                results.append({
                    "source": doc.source,
                    "chunks": 0,
                    "status": "error",
                    "error": str(e),
                })
        
        return {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "results": results,
        }
    
    async def ingest_text(
        self,
        text: str,
        source: str = "manual_input",
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest raw text directly.
        
        Useful for:
        - API uploads
        - Manual text entry
        - Testing
        
        Args:
            text: Text content to ingest
            source: Source identifier
            metadata: Optional metadata
            
        Returns:
            Ingestion result
        """
        from .chunker import Document
        
        document = Document(
            content=text,
            source=source,
            doc_type="text",
            metadata=metadata or {},
        )
        
        return await self.ingest_document(document)
