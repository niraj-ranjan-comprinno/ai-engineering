"""
Embedding Service using AWS Bedrock Titan.

=============================================================================
THEORY: What are Embeddings?
=============================================================================

Embeddings are dense vector representations of text that capture semantic meaning.

Text: "The cat sat on the mat"
           ↓ Embedding Model
Vector: [0.023, -0.156, 0.892, ..., 0.045]  (1024 dimensions for Titan V2)

Why Embeddings Matter for RAG:
------------------------------
1. Semantic Search: "dog" and "puppy" have similar vectors (unlike keyword search)
2. Efficient Comparison: Cosine similarity is fast on vectors
3. Language Agnostic: Similar concepts cluster regardless of language

How It Works:
-------------
1. Neural network (transformer) processes text
2. Outputs a fixed-size vector (e.g., 1024 dimensions)
3. Similar meanings → Similar vectors → Close in vector space

Distance/Similarity:
--------------------
Cosine Similarity: cos(θ) = (A · B) / (||A|| × ||B||)
- 1.0 = identical
- 0.0 = unrelated
- -1.0 = opposite

Example:
    embed("happy") ≈ embed("joyful")     → similarity ~0.9
    embed("happy") ≈ embed("computer")   → similarity ~0.2
    embed("happy") ≈ embed("sad")        → similarity ~0.3

Embedding Models:
-----------------
| Model | Dimensions | Context | Provider |
|-------|------------|---------|----------|
| Titan V2 | 1024 | 8192 tokens | AWS Bedrock |
| text-embedding-3-large | 3072 | 8191 tokens | OpenAI |
| text-embedding-3-small | 1536 | 8191 tokens | OpenAI |
| Cohere embed-v3 | 1024 | 512 tokens | Cohere |

=============================================================================
"""

import asyncio
import json
import logging
from typing import Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


class BedrockEmbeddings:
    """
    Generate embeddings using AWS Bedrock Titan Embeddings.
    
    THEORY: Bedrock Titan Embeddings
    --------------------------------
    Amazon Titan Text Embeddings V2:
    - 1024 dimensions (configurable: 256, 512, 1024)
    - 8192 token context window
    - Supports multiple languages
    - Normalized vectors (unit length)
    
    Pricing (as of 2024):
    - $0.00002 per 1K input tokens
    - Very cost-effective for embeddings
    
    Usage:
    - Input: Text string
    - Output: List of floats (embedding vector)
    """
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        profile_name: Optional[str] = None,
        dimensions: int = 1024,
    ):
        """
        Initialize Bedrock embeddings client.
        
        Args:
            region_name: AWS region
            model_id: Bedrock model ID for embeddings
            profile_name: AWS CLI profile name
            dimensions: Output embedding dimensions (256, 512, or 1024)
        """
        config = Config(
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        session_kwargs = {}
        if profile_name:
            session_kwargs['profile_name'] = profile_name
        
        session = boto3.Session(**session_kwargs)
        
        self._client = session.client(
            'bedrock-runtime',
            region_name=region_name,
            config=config,
        )
        self._model_id = model_id
        self._dimensions = dimensions
        
        logger.info(
            f"BedrockEmbeddings initialized: model={model_id}, "
            f"dimensions={dimensions}, region={region_name}"
        )
    
    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions
    
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (max 8192 tokens)
            
        Returns:
            List of floats (embedding vector)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Titan V2 request format
        body = {
            "inputText": text,
            "dimensions": self._dimensions,
            "normalize": True,  # Unit-length vectors for cosine similarity
        }
        
        # Run synchronous boto3 in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(
                modelId=self._model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        )
        
        result = json.loads(response['body'].read())
        embedding = result['embedding']
        
        logger.debug(f"Generated embedding: {len(embedding)} dimensions")
        return embedding
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        THEORY: Batching Embeddings
        ---------------------------
        For efficiency, process multiple texts together.
        Options:
        1. Sequential: Simple but slow
        2. Concurrent: Use asyncio.gather for parallel API calls
        3. True Batch: Some APIs support batch endpoints
        
        We use concurrent processing here (option 2).
        """
        if not texts:
            return []
        
        # Process concurrently
        tasks = [self.embed_text(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return list(embeddings)
    
    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        
        THEORY: Query vs Document Embeddings
        ------------------------------------
        Some embedding models (like Cohere) distinguish between:
        - Document embeddings: For storing in the database
        - Query embeddings: For searching
        
        Titan V2 doesn't require this distinction, but we keep
        the method for API consistency and future flexibility.
        """
        return await self.embed_text(query)


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    THEORY: Cosine Similarity
    -------------------------
    Measures the angle between two vectors, ignoring magnitude.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Range: -1 to 1
    - 1.0 = identical direction
    - 0.0 = orthogonal (unrelated)
    - -1.0 = opposite direction
    
    For normalized vectors (unit length), this simplifies to dot product.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitudes
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    THEORY: Euclidean Distance
    --------------------------
    Measures the "straight line" distance between two points.
    
    Formula: d = √(Σ(a_i - b_i)²)
    
    Range: 0 to ∞
    - 0 = identical
    - Higher = more different
    
    Less commonly used for embeddings (cosine is preferred).
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
