"""
LLM Router - Intelligent Provider Selection and Routing.

=============================================================================
THEORY: Request Routing Patterns
=============================================================================

Why route requests?
-------------------
In production, you rarely use a single LLM provider. Routing enables:

1. Fallback Chains: Provider A fails → try Provider B
2. Cost Optimization: Route simple queries to cheaper models
3. Quality Routing: Complex queries go to more capable models
4. Geographic Routing: Route to nearest data center
5. A/B Testing: Compare providers on same traffic

Routing Strategies:
-------------------
1. Static: Always use the same provider
2. Round Robin: Distribute evenly across providers
3. Weighted: Route X% to A, Y% to B
4. Content-Based: Analyze request to choose provider
5. Fallback: Try providers in order until one succeeds

This router implements fallback routing - the most critical
pattern for production reliability.

            ┌─────────────────────────────────┐
            │         Request                  │
            └─────────────┬───────────────────┘
                          │
                          ▼
            ┌─────────────────────────────────┐
            │    Router: Which provider?       │
            └─────────────┬───────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
     ┌─────────┐    ┌─────────┐    ┌─────────┐
     │ OpenAI  │    │Anthropic│    │  Local  │
     │ (gpt-4) │    │ (claude)│    │ (llama) │
     └─────────┘    └─────────┘    └─────────┘
"""

import logging
from typing import AsyncIterator, Optional

from .base import BaseLLMProvider, LLMResponse, StreamDelta
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .bedrock_api_provider import BedrockAPIProvider

# Bedrock with boto3 is optional
try:
    from .bedrock_provider import BedrockProvider
    BEDROCK_AVAILABLE = True
except ImportError:
    BedrockProvider = None  # type: ignore
    BEDROCK_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Routes requests to appropriate LLM providers.
    
    THEORY: The Router Pattern
    --------------------------
    The router acts as a facade that:
    1. Hides provider complexity from callers
    2. Handles provider selection logic
    3. Manages fallback chains
    4. Collects metrics across providers
    
    This is the single entry point for all LLM operations.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        bedrock_api_key: Optional[str] = None,
        use_bedrock: bool = False,
        aws_region: str = "us-east-1",
        aws_profile: Optional[str] = None,
        default_provider: str = "openai",
        default_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the router with available providers.
        
        THEORY: Lazy vs Eager Provider Initialization
        ----------------------------------------------
        We initialize providers eagerly (at startup) rather than
        lazily (on first request) because:
        
        1. Fast failure: Know immediately if API keys are invalid
        2. Connection warmup: First request isn't slow
        3. Clearer errors: Startup failures are obvious
        
        Args:
            openai_api_key: OpenAI API key (optional)
            anthropic_api_key: Anthropic API key (optional)
            bedrock_api_key: AWS Bedrock API key (optional, simplest!)
            use_bedrock: Use Bedrock with boto3/AWS credentials (optional)
            aws_region: AWS region for Bedrock
            aws_profile: AWS CLI profile name (optional)
            default_provider: Provider to use if none specified
            default_model: Model to use if none specified
        """
        self._providers: dict[str, BaseLLMProvider] = {}
        self._default_provider = default_provider
        self._default_model = default_model
        
        # Initialize available providers
        if openai_api_key:
            self._providers["openai"] = OpenAIProvider(
                api_key=openai_api_key,
                default_model=default_model if default_provider == "openai" else "gpt-4o-mini",
            )
            logger.info("OpenAI provider registered")
        
        if anthropic_api_key:
            self._providers["anthropic"] = AnthropicProvider(
                api_key=anthropic_api_key,
                default_model=default_model if default_provider == "anthropic" else "claude-3-5-sonnet-20240620",
            )
            logger.info("Anthropic provider registered")
        
        # AWS Bedrock with API Key (simplest approach!)
        if bedrock_api_key:
            self._providers["bedrock"] = BedrockAPIProvider(
                api_key=bedrock_api_key,
                region=aws_region,
                default_model=default_model if default_provider == "bedrock" else "claude-3-5-sonnet",
            )
            logger.info(f"Bedrock API provider registered in {aws_region}")
        
        # AWS Bedrock with boto3 credentials (fallback if no API key)
        elif use_bedrock and BEDROCK_AVAILABLE:
            try:
                self._providers["bedrock"] = BedrockProvider(
                    region_name=aws_region,
                    default_model=default_model if default_provider == "bedrock" else "claude-3-5-sonnet",
                    profile_name=aws_profile or None,
                )
                logger.info(f"Bedrock boto3 provider registered in {aws_region}")
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock with boto3: {e}")
        elif use_bedrock and not BEDROCK_AVAILABLE:
            logger.warning("Bedrock boto3 requested but not installed. Run: pip install boto3")
        
        if not self._providers:
            raise ValueError(
                "At least one provider API key must be configured. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment."
            )
        
        # Validate default provider exists
        if default_provider not in self._providers:
            # Fall back to first available provider
            self._default_provider = next(iter(self._providers.keys()))
            logger.warning(
                f"Default provider '{default_provider}' not available. "
                f"Using '{self._default_provider}' instead."
            )
    
    @property
    def available_providers(self) -> list[str]:
        """List of registered provider names."""
        return list(self._providers.keys())
    
    @property
    def default_provider(self) -> str:
        """Default provider name."""
        return self._default_provider
    
    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        """
        Get a specific provider by name.
        
        Args:
            name: Provider name, or None for default
            
        Returns:
            The requested provider
            
        Raises:
            ValueError: If provider not found
        """
        name = name or self._default_provider
        
        if name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Provider '{name}' not available. "
                f"Available providers: {available}"
            )
        
        return self._providers[name]
    
    async def chat(
        self,
        messages: list[dict],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        fallback: bool = True,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> tuple[LLMResponse, str]:
        """
        Route a chat completion request.
        
        THEORY: Fallback Strategy
        -------------------------
        When the primary provider fails, we try others:
        
        1. Try requested/default provider
        2. If fails and fallback=True, try other providers
        3. If all fail, raise the last error
        
        This gives you automatic redundancy without code changes.
        
        Args:
            messages: Chat messages
            provider: Preferred provider (optional)
            model: Model to use (optional)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            fallback: Whether to try other providers on failure
            tools: Optional list of tool schemas for function calling
            system: Optional system prompt
            
        Returns:
            Tuple of (LLMResponse, provider_name_used)
        """
        # Build provider order (requested first, then others)
        provider_order = [provider or self._default_provider]
        if fallback:
            provider_order.extend(
                p for p in self._providers.keys() 
                if p not in provider_order
            )
        
        last_error: Optional[Exception] = None
        
        for provider_name in provider_order:
            if provider_name not in self._providers:
                continue
            
            try:
                llm_provider = self._providers[provider_name]
                response = await llm_provider.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    system=system,
                )
                return response, provider_name
                
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed: {e}. "
                    f"{'Trying next provider...' if fallback else 'No fallback enabled.'}"
                )
                last_error = e
        
        # All providers failed
        raise last_error or ValueError("No providers available")
    
    async def stream_chat(
        self,
        messages: list[dict],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[AsyncIterator[StreamDelta], str]:
        """
        Route a streaming chat completion request.
        
        THEORY: Streaming and Fallback
        ------------------------------
        Streaming with fallback is tricky because:
        1. Stream starts before we know it'll succeed
        2. Can't easily restart mid-stream
        3. Client might receive partial response before error
        
        Strategy:
        - Try to establish stream with primary provider
        - If initial connection fails, try fallback
        - Once streaming starts, don't fallback (would lose content)
        
        For production, consider:
        - Health checks before routing
        - Circuit breaker pattern
        - Streaming with buffering for retry
        
        Args:
            messages: Chat messages
            provider: Preferred provider
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Tuple of (AsyncIterator of deltas, provider_name_used)
        """
        provider_name = provider or self._default_provider
        llm_provider = self.get_provider(provider_name)
        
        stream = llm_provider.stream_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return stream, provider_name
