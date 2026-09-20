"""
Base Provider Interface - The Contract All Providers Must Follow.

=============================================================================
THEORY: Abstract Base Classes (ABCs) in Python
=============================================================================

Abstract Base Classes define a contract that subclasses MUST implement.
This is Python's way of doing interfaces.

Why use ABCs?
1. Enforced contracts: Python raises TypeError if methods aren't implemented
2. Clear documentation: The interface shows exactly what's expected
3. IDE support: Type hints enable autocomplete and error detection
4. Runtime safety: Errors at instantiation, not at method call

AsyncIterator and Async Generators:
-----------------------------------
For streaming, we use async generators (async def + yield).
This allows:
- Processing chunks as they arrive (no buffering entire response)
- Non-blocking I/O (other requests can be served while waiting)
- Memory efficiency (don't store entire response in memory)

The pattern:
    async for chunk in provider.stream_chat(request):
        yield chunk  # Send to client immediately
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class LLMResponse:
    """
    Unified response format from any LLM provider.
    
    Using a common response format means:
    - Application code doesn't care which provider was used
    - Easy to log and compare across providers
    - Consistent metrics collection
    """
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StreamDelta:
    """
    A single chunk in a streaming response.
    
    THEORY: Delta vs Full Content
    -----------------------------
    Streaming returns DELTAS (changes), not full content:
    
    Chunk 1: delta="Hello"      full_content="Hello"
    Chunk 2: delta=", "         full_content="Hello, "
    Chunk 3: delta="world"      full_content="Hello, world"
    Chunk 4: delta="!"          full_content="Hello, world!"
    
    The client accumulates deltas to build the full response.
    This is memory-efficient for the server (doesn't track state).
    """
    content: str
    finish_reason: Optional[str] = None
    # Token counts only available in final chunk (for OpenAI)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    THEORY: The Provider Interface
    ------------------------------
    Every LLM provider (OpenAI, Anthropic, Cohere, local models, etc.)
    MUST implement these methods:
    
    1. chat() - Synchronous completion (wait for full response)
    2. stream_chat() - Streaming completion (yield chunks)
    3. model property - List of available models
    
    This abstraction lets us:
    - Add new providers without changing application code
    - Test with mock providers
    - Implement provider-specific optimizations internally
    
    The @abstractmethod decorator means Python will raise TypeError
    if you try to instantiate a class that doesn't implement these.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'openai', 'anthropic')."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model to use if none specified."""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> list[str]:
        """List of models this provider supports."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a complete response (non-streaming).
        
        THEORY: When to use non-streaming
        ---------------------------------
        Non-streaming is simpler and better for:
        - Background processing (no user waiting)
        - Batch operations
        - When you need the full response before proceeding
        - Function calling / tool use (need complete JSON)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (provider-specific)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with content and usage stats
        """
        pass
    
    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamDelta]:
        """
        Generate a streaming response.
        
        THEORY: Async Generators
        ------------------------
        This method is an async generator - it uses 'yield' instead of 'return'.
        
        async def stream_chat(...) -> AsyncIterator[StreamDelta]:
            async for chunk in api_response:
                yield StreamDelta(content=chunk.text)
        
        The caller consumes it with async for:
        
            async for delta in provider.stream_chat(messages):
                print(delta.content, end="")
        
        Benefits:
        - Memory efficient: Only one chunk in memory at a time
        - Low latency: First chunk sent immediately
        - Backpressure: Generator pauses if consumer is slow
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            StreamDelta objects with content chunks
        """
        pass
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a given model."""
        return model in self.available_models
