"""
OpenAI Provider Implementation.

=============================================================================
THEORY: OpenAI API Patterns
=============================================================================

OpenAI's Chat Completions API:
- Most widely used LLM API
- Established the messages format that others follow
- Supports streaming via Server-Sent Events

Key Concepts:
1. Messages format: [{"role": "...", "content": "..."}]
2. Streaming: chunks arrive as delta.content
3. Usage tracking: Available in stream_options or final response

The OpenAI Python SDK:
- AsyncOpenAI for async operations (FastAPI needs this)
- Automatic retries with exponential backoff
- Proper error handling with typed exceptions

IMPORTANT: Always use async client with async frameworks!
Sync client blocks the event loop = terrible performance.
"""

import logging
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APIConnectionError

from .base import BaseLLMProvider, LLMResponse, StreamDelta

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider implementation.
    
    THEORY: Client Lifecycle
    ------------------------
    The OpenAI client should be:
    1. Created once at startup (reuse connections)
    2. Shared across requests (connection pooling)
    3. NOT created per-request (expensive!)
    
    httpx (used internally) maintains a connection pool,
    so reusing the client means:
    - No TCP handshake per request
    - No TLS negotiation per request
    - Much lower latency
    """
    
    # Models we officially support (add more as needed)
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
    ]
    
    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (starts with 'sk-')
            default_model: Model to use if none specified in request
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._default_model = default_model
        logger.info(f"OpenAI provider initialized with default model: {default_model}")
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def default_model(self) -> str:
        return self._default_model
    
    @property
    def available_models(self) -> list[str]:
        return self.SUPPORTED_MODELS
    
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a non-streaming response from OpenAI.
        
        THEORY: Error Handling Strategy
        --------------------------------
        LLM APIs fail in predictable ways:
        
        1. RateLimitError (429): Too many requests
           → Implement exponential backoff + retry
           → Consider multiple API keys
        
        2. APIConnectionError: Network issues
           → Retry with backoff
           → Consider fallback provider
        
        3. APIError (400, 500, etc.): Bad request or server error
           → 400: Check your request format
           → 500: Retry, then fallback
        
        The SDK handles retries automatically for transient errors.
        We catch and re-raise with context for better debugging.
        """
        model = model or self._default_model
        
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract response data
            choice = response.choices[0]
            usage = response.usage
            
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason or "stop",
            )
            
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamDelta]:
        """
        Generate a streaming response from OpenAI.
        
        THEORY: OpenAI Streaming Format
        --------------------------------
        OpenAI streaming returns chunks like this:
        
        data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}
        data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}
        data: {"choices":[{"delta":{"content":" world"},"index":0}]}
        data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
        data: [DONE]
        
        Notice:
        1. First chunk has role but no content
        2. Middle chunks have content in delta
        3. Final chunk has finish_reason and empty delta
        4. [DONE] signals end of stream
        
        stream_options={"include_usage": True} adds token counts to final chunk.
        This is CRITICAL for cost tracking in streaming mode!
        """
        model = model or self._default_model
        
        try:
            # Enable usage tracking in stream
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},  # Get token counts!
            )
            
            async for chunk in stream:
                # Skip empty chunks
                if not chunk.choices:
                    # Final chunk with usage stats
                    if chunk.usage:
                        yield StreamDelta(
                            content="",
                            finish_reason="stop",
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Extract content (might be None for first/last chunks)
                content = delta.content or ""
                
                yield StreamDelta(
                    content=content,
                    finish_reason=choice.finish_reason,
                )
                
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded during stream: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error during stream: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error during stream: {e}")
            raise
