"""
Pydantic Models for Request/Response Schemas.

=============================================================================
THEORY: Data Validation with Pydantic
=============================================================================

Pydantic provides runtime data validation using Python type hints.
This is CRITICAL for AI applications because:

1. LLM APIs are strict - wrong types = failed requests = wasted money
2. Type safety catches bugs before they reach production
3. Auto-generated API documentation (OpenAPI/Swagger)
4. Clear contracts between client and server

Key Concepts:
-------------
- BaseModel: All schemas inherit from this
- Field(): Adds validation, defaults, and documentation
- Literal: Restricts values to specific options
- Optional: Field can be None
"""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class Message(BaseModel):
    """
    A single message in a conversation.
    
    THEORY: Chat Completion Message Format
    --------------------------------------
    LLMs use a role-based message format:
    - "system": Instructions that guide the model's behavior
    - "user": The human's input
    - "assistant": The model's previous responses
    
    This structure enables:
    1. Multi-turn conversations (context is maintained)
    2. Persona customization via system prompts
    3. Few-shot learning by providing example exchanges
    """
    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message author"
    )
    content: str = Field(
        description="The content of the message",
        min_length=1,
    )


class ChatRequest(BaseModel):
    """
    Request body for chat completions.
    
    THEORY: Temperature and Top-P Sampling
    ---------------------------------------
    These parameters control the "randomness" of model outputs:
    
    Temperature (0.0 - 2.0):
    - Lower = more deterministic, focused responses
    - Higher = more creative, diverse responses
    - 0.0 = always pick the most likely next token
    - 1.0 = sample according to probability distribution
    
    Top-P (Nucleus Sampling, 0.0 - 1.0):
    - Only consider tokens in the top P probability mass
    - 0.1 = only consider top 10% most likely tokens
    - 1.0 = consider all tokens
    
    Best Practice: Adjust temperature OR top_p, not both simultaneously.
    """
    messages: list[Message] = Field(
        description="List of messages in the conversation",
        min_length=1,
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use (defaults to server default)",
    )
    provider: Optional[Literal["openai", "anthropic", "bedrock"]] = Field(
        default=None,
        description="LLM provider to use (defaults to server default)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=128000,
        description="Maximum tokens to generate",
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream the response",
    )


# =============================================================================
# Response Models
# =============================================================================

class TokenUsage(BaseModel):
    """
    Token usage statistics for a request.
    
    THEORY: Understanding Tokens
    ----------------------------
    Tokens are the fundamental units LLMs work with. They're NOT words!
    
    Examples (using GPT tokenizer):
    - "Hello" = 1 token
    - "Hello, world!" = 4 tokens  
    - "Pneumonoultramicroscopicsilicovolcanoconiosis" = 10 tokens
    - Code often has MORE tokens than equivalent English
    
    Why tokens matter:
    1. Billing is per-token (input + output)
    2. Context window limits are in tokens
    3. Latency correlates with token count
    
    Token Economics:
    - Input tokens: Usually cheaper
    - Output tokens: Usually 2-4x more expensive
    - Long prompts = higher cost + slower response
    """
    prompt_tokens: int = Field(description="Tokens in the input prompt")
    completion_tokens: int = Field(description="Tokens in the generated response")
    total_tokens: int = Field(description="Total tokens used")


class CostBreakdown(BaseModel):
    """
    Cost breakdown for a request.
    
    THEORY: LLM Cost Optimization
    -----------------------------
    Understanding costs is crucial for production AI systems:
    
    Cost = (input_tokens × input_price) + (output_tokens × output_price)
    
    Optimization strategies:
    1. Use smaller models when possible (gpt-4o-mini vs gpt-4)
    2. Reduce prompt length (shorter system prompts)
    3. Limit max_tokens for short-answer tasks
    4. Cache common responses
    5. Use embeddings for similarity instead of full LLM calls
    """
    input_cost: float = Field(description="Cost of input tokens in USD")
    output_cost: float = Field(description="Cost of output tokens in USD")
    total_cost: float = Field(description="Total cost in USD")
    model: str = Field(description="Model used for pricing")


class ChatResponse(BaseModel):
    """Complete response for non-streaming requests."""
    id: str = Field(description="Unique request identifier")
    content: str = Field(description="Generated response content")
    model: str = Field(description="Model used for generation")
    provider: str = Field(description="Provider used")
    usage: TokenUsage = Field(description="Token usage statistics")
    cost: Optional[CostBreakdown] = Field(
        default=None,
        description="Cost breakdown (if tracking enabled)",
    )
    latency_ms: float = Field(description="Request latency in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StreamChunk(BaseModel):
    """
    A single chunk in a streaming response.
    """
    id: str = Field(description="Request identifier")
    delta: str = Field(description="New content in this chunk")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Why generation stopped (only in final chunk)",
    )


class StreamComplete(BaseModel):
    """Final message in a stream with usage stats."""
    id: str
    usage: TokenUsage
    cost: Optional[CostBreakdown] = None
    latency_ms: float
    model: str
    provider: str


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
