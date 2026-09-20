"""
LLM Provider Abstraction Layer.

=============================================================================
THEORY: Provider Abstraction Pattern
=============================================================================

Why abstract LLM providers?
---------------------------
1. Vendor Independence: Swap providers without changing application code
2. Fallbacks: Automatically switch to backup provider if primary fails
3. A/B Testing: Route traffic to different providers for comparison
4. Cost Optimization: Route to cheapest provider based on request type
5. Rate Limit Management: Spread load across multiple providers

Design Pattern: Strategy Pattern
--------------------------------
We define a common interface (BaseLLMProvider) and each provider
implements it. The application code depends on the interface, not
the concrete implementation.

    ChatRequest
         │
         ▼
    ┌─────────────────┐
    │  LLMRouter      │ ◄── Decides which provider to use
    └────────┬────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐     ┌─────────┐
│ OpenAI  │     │Anthropic│
└─────────┘     └─────────┘

This is foundational for building AI systems that are:
- Reliable (provider outages don't break your app)
- Cost-effective (route to optimal provider)
- Future-proof (new providers are easy to add)
"""

from .base import BaseLLMProvider, LLMResponse, StreamDelta
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .router import LLMRouter

# Bedrock is optional (requires boto3)
try:
    from .bedrock_provider import BedrockProvider
    BEDROCK_AVAILABLE = True
except ImportError:
    BedrockProvider = None  # type: ignore
    BEDROCK_AVAILABLE = False

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "StreamDelta",
    "OpenAIProvider",
    "AnthropicProvider",
    "BedrockProvider",
    "LLMRouter",
    "BEDROCK_AVAILABLE",
]
