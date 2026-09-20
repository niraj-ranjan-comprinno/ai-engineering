"""
FastAPI Routes for the AI Gateway.

=============================================================================
THEORY: Server-Sent Events (SSE) for Streaming
=============================================================================

What is SSE?
------------
Server-Sent Events is a standard for pushing data from server to client
over a single HTTP connection. Unlike WebSockets, it's:
- Unidirectional (server → client only)
- Uses standard HTTP (works through proxies/firewalls)
- Auto-reconnects on connection loss
- Native browser support via EventSource API

SSE Format:
-----------
Each event is text with this format:
    data: {"content": "Hello"}\n\n
    data: {"content": " World"}\n\n
    data: [DONE]\n\n

The double newline (\n\n) marks the end of each event.
Multiple `data:` lines are concatenated with newlines.

Why SSE for LLM Streaming?
--------------------------
1. Perfect fit: Server pushes tokens as generated
2. Simple: No WebSocket complexity
3. Works everywhere: HTTP-based, proxy-friendly
4. Efficient: Single connection, no polling

Alternative: WebSockets
-----------------------
Use WebSockets when you need:
- Bidirectional communication
- Very low latency (gaming, real-time collab)
- Binary data transfer

For LLM chat, SSE is the better choice.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..config import Settings, get_settings
from ..models import (
    ChatRequest,
    ChatResponse,
    CostBreakdown,
    ErrorResponse,
    StreamChunk,
    StreamComplete,
    TokenUsage,
)
from ..providers import LLMRouter
from ..tokenizer import calculate_cost, count_message_tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


# =============================================================================
# Dependency Injection
# =============================================================================

def get_llm_router(settings: Settings = Depends(get_settings)) -> LLMRouter:
    """
    Create LLM router with configured providers.
    """
    return LLMRouter(
        openai_api_key=settings.openai_api_key or None,
        anthropic_api_key=settings.anthropic_api_key or None,
        bedrock_api_key=settings.bedrock_api_key or None,
        use_bedrock=settings.use_bedrock,
        aws_region=settings.aws_region,
        aws_profile=settings.aws_profile or None,
        default_provider=settings.default_provider,
        default_model=settings.default_model,
    )


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    THEORY: Health Checks in Production
    ------------------------------------
    Health checks are critical for:
    1. Load balancer routing (only route to healthy instances)
    2. Kubernetes liveness/readiness probes
    3. Monitoring and alerting
    
    Types of health checks:
    - Shallow: "Am I running?" (this one)
    - Deep: "Can I connect to dependencies?"
    
    For production, add:
    - Provider connectivity checks
    - Database connection check
    - Memory/CPU usage
    """
    return {"status": "healthy", "service": "ai-gateway"}


# =============================================================================
# Provider Info
# =============================================================================

@router.get("/providers")
async def list_providers(
    llm_router: LLMRouter = Depends(get_llm_router),
):
    """
    List available LLM providers and their models.
    
    Useful for:
    - Client discovery of available options
    - Debugging configuration issues
    - Building dynamic UIs
    """
    providers_info = {}
    for provider_name in llm_router.available_providers:
        provider = llm_router.get_provider(provider_name)
        providers_info[provider_name] = {
            "models": provider.available_models,
            "default_model": provider.default_model,
        }
    
    return {
        "providers": providers_info,
        "default_provider": llm_router.default_provider,
    }


# =============================================================================
# Chat Completions (Non-Streaming)
# =============================================================================

@router.post(
    "/chat/completions",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat_completions(
    request: ChatRequest,
    llm_router: LLMRouter = Depends(get_llm_router),
    settings: Settings = Depends(get_settings),
):
    """
    Generate a chat completion.
    
    THEORY: Request Lifecycle
    -------------------------
    1. Request arrives, Pydantic validates body
    2. Dependencies are resolved (settings, router)
    3. Business logic executes
    4. Response is serialized and returned
    
    Error handling strategy:
    - Pydantic errors → 422 Unprocessable Entity (automatic)
    - Business logic errors → 400 Bad Request
    - Provider errors → 500 Internal Server Error
    - Rate limits → 429 Too Many Requests
    
    If streaming is requested, we delegate to SSE endpoint.
    """
    # If streaming requested, use the streaming endpoint
    if request.stream:
        return await stream_chat_completions(request, llm_router, settings)
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Convert messages to dict format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Call LLM
        response, provider_used = await llm_router.chat(
            messages=messages,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate cost if enabled
        cost = None
        if settings.enable_cost_tracking:
            cost_info = calculate_cost(
                response.prompt_tokens,
                response.completion_tokens,
                response.model,
            )
            cost = CostBreakdown(
                input_cost=cost_info["input_cost"],
                output_cost=cost_info["output_cost"],
                total_cost=cost_info["total_cost"],
                model=response.model,
            )
        
        return ChatResponse(
            id=request_id,
            content=response.content,
            model=response.model,
            provider=provider_used,
            usage=TokenUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            ),
            cost=cost,
            latency_ms=round(latency_ms, 2),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# =============================================================================
# Streaming Chat Completions (SSE)
# =============================================================================

async def stream_chat_completions(
    request: ChatRequest,
    llm_router: LLMRouter,
    settings: Settings,
) -> EventSourceResponse:
    """
    Stream a chat completion using Server-Sent Events.
    
    THEORY: Async Generators for Streaming
    --------------------------------------
    Python's async generators (async def + yield) are perfect for streaming:
    
        async def generate():
            async for chunk in source:
                yield format_sse(chunk)
    
    The FastAPI/Starlette framework handles:
    1. Keeping the HTTP connection open
    2. Sending chunks as they're yielded
    3. Proper SSE formatting
    4. Connection cleanup on client disconnect
    
    Client-side consumption:
        const eventSource = new EventSource('/v1/chat/stream');
        eventSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            console.log(data.delta);
        };
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        """
        Generate SSE events from LLM stream.
        
        THEORY: Event Types in SSE
        --------------------------
        SSE supports named events:
            event: chunk
            data: {"delta": "Hello"}
            
        We use:
        - "chunk": Content deltas during generation
        - "complete": Final message with usage stats
        - "error": Error notification
        
        The `data` field contains JSON with our payload.
        """
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        model_used = request.model or settings.default_model
        provider_used = request.provider or settings.default_provider
        
        try:
            # Convert messages
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            
            # Get stream from provider
            stream, provider_used = await llm_router.stream_chat(
                messages=messages,
                provider=request.provider,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            # Stream chunks
            async for delta in stream:
                if delta.content:
                    full_content += delta.content
                    
                    # Yield content chunk
                    chunk = StreamChunk(
                        id=request_id,
                        delta=delta.content,
                        finish_reason=None,
                    )
                    yield {
                        "event": "chunk",
                        "data": chunk.model_dump_json(),
                    }
                
                # Capture token counts from final delta
                if delta.prompt_tokens is not None:
                    prompt_tokens = delta.prompt_tokens
                if delta.completion_tokens is not None:
                    completion_tokens = delta.completion_tokens
                
                # Final chunk with finish reason
                if delta.finish_reason:
                    chunk = StreamChunk(
                        id=request_id,
                        delta="",
                        finish_reason=delta.finish_reason,
                    )
                    yield {
                        "event": "chunk",
                        "data": chunk.model_dump_json(),
                    }
            
            # Calculate final metrics
            latency_ms = (time.time() - start_time) * 1000
            
            # If we didn't get token counts from stream, estimate them
            if prompt_tokens == 0:
                prompt_tokens = count_message_tokens(messages, model_used)
            if completion_tokens == 0:
                # Rough estimate: count tokens in response
                from ..tokenizer import count_tokens
                completion_tokens = count_tokens(full_content, model_used)
            
            # Calculate cost
            cost = None
            if settings.enable_cost_tracking:
                cost_info = calculate_cost(prompt_tokens, completion_tokens, model_used)
                cost = CostBreakdown(
                    input_cost=cost_info["input_cost"],
                    output_cost=cost_info["output_cost"],
                    total_cost=cost_info["total_cost"],
                    model=model_used,
                )
            
            # Send completion event with stats
            complete = StreamComplete(
                id=request_id,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                cost=cost,
                latency_ms=round(latency_ms, 2),
                model=model_used,
                provider=provider_used,
            )
            yield {
                "event": "complete",
                "data": complete.model_dump_json(),
            }
            
            # Signal end of stream
            yield {
                "event": "done",
                "data": "[DONE]",
            }
            
        except Exception as e:
            logger.exception(f"Stream error: {e}")
            error_data = json.dumps({
                "error": type(e).__name__,
                "message": str(e),
            })
            yield {
                "event": "error",
                "data": error_data,
            }
    
    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# =============================================================================
# Token Counting Utility
# =============================================================================

@router.post("/tokens/count")
async def count_tokens_endpoint(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
):
    """
    Count tokens for a message without making an LLM call.
    
    THEORY: Pre-flight Token Counting
    ---------------------------------
    Why count tokens before the actual call?
    
    1. Cost Estimation: Know the cost before committing
    2. Context Management: Check if messages fit in context window
    3. Prompt Optimization: A/B test prompt lengths
    4. Budgeting: Implement token budgets per user/request
    
    Use cases:
    - "Will this fit in 4096 tokens?"
    - "How much will this request cost?"
    - "Should I truncate the conversation history?"
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    model = request.model or settings.default_model
    
    token_count = count_message_tokens(messages, model)
    
    # Estimate cost if max_tokens is specified
    cost_estimate = None
    if request.max_tokens and settings.enable_cost_tracking:
        # Estimate: assume we'll use all max_tokens
        cost_info = calculate_cost(token_count, request.max_tokens, model)
        cost_estimate = {
            "estimated_input_cost": cost_info["input_cost"],
            "estimated_max_output_cost": cost_info["output_cost"],
            "estimated_max_total_cost": cost_info["total_cost"],
        }
    
    return {
        "prompt_tokens": token_count,
        "model": model,
        "cost_estimate": cost_estimate,
    }
