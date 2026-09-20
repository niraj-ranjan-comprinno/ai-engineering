"""
Anthropic Claude Provider Implementation.

=============================================================================
THEORY: Anthropic API Differences from OpenAI
=============================================================================

Anthropic's API has key differences:

1. System Message Handling:
   - OpenAI: system message in messages array
   - Anthropic: separate 'system' parameter
   
2. Message Format:
   - OpenAI: {"role": "user", "content": "Hello"}
   - Anthropic: {"role": "user", "content": "Hello"} (same!)
   But Anthropic requires alternating user/assistant messages.

3. Streaming Events:
   - OpenAI: Single event type with delta
   - Anthropic: Multiple event types (message_start, content_block_delta, etc.)

4. Token Counting:
   - OpenAI: In response
   - Anthropic: In message_start and message_delta events

These differences are why we have the provider abstraction!
Application code doesn't need to know these details.
"""

import logging
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic
from anthropic import APIError, RateLimitError, APIConnectionError

from .base import BaseLLMProvider, LLMResponse, StreamDelta

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider implementation.
    
    THEORY: Claude's Strengths
    --------------------------
    Claude models excel at:
    - Long context (200K tokens!)
    - Following complex instructions
    - Nuanced, thoughtful responses
    - Avoiding harmful outputs
    
    Trade-offs vs GPT-4:
    - Generally more verbose
    - Different pricing structure
    - Sometimes more cautious/refusing
    """
    
    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-latest",
        "claude-3-opus-20240229",
        "claude-3-opus-latest",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-3-5-sonnet-20240620"
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (starts with 'sk-ant-')
            default_model: Model to use if none specified
        """
        self._client = AsyncAnthropic(api_key=api_key)
        self._default_model = default_model
        logger.info(f"Anthropic provider initialized with default model: {default_model}")
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def default_model(self) -> str:
        return self._default_model
    
    @property
    def available_models(self) -> list[str]:
        return self.SUPPORTED_MODELS
    
    def _extract_system_message(
        self,
        messages: list[dict]
    ) -> tuple[Optional[str], list[dict]]:
        """
        Extract system message from messages list.
        
        THEORY: Anthropic System Message Pattern
        ----------------------------------------
        Anthropic's API takes system as a separate parameter:
        
        OpenAI style:
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"}
            ]
        
        Anthropic style:
            system="You are helpful"
            messages=[
                {"role": "user", "content": "Hi"}
            ]
        
        We convert automatically so application code can use OpenAI format.
        """
        system_content = None
        filtered_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                # Concatenate multiple system messages
                if system_content:
                    system_content += "\n\n" + msg["content"]
                else:
                    system_content = msg["content"]
            else:
                filtered_messages.append(msg)
        
        return system_content, filtered_messages
    
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a non-streaming response from Anthropic.
        
        THEORY: max_tokens Behavior
        ---------------------------
        Anthropic REQUIRES max_tokens (unlike OpenAI where it's optional).
        
        Best practices:
        - Set a reasonable default (4096 is safe for most tasks)
        - Match to your use case (longer for creative, shorter for Q&A)
        - Remember: more max_tokens = potentially higher cost
        """
        model = model or self._default_model
        max_tokens = max_tokens or 4096
        
        # Extract system message
        system_content, filtered_messages = self._extract_system_message(messages)
        
        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=filtered_messages,  # type: ignore
                system=system_content or "",
                temperature=temperature,
            )
            
            # Anthropic returns content as a list of blocks
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
            
            return LLMResponse(
                content=content,
                model=response.model,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                finish_reason=response.stop_reason or "stop",
            )
            
        except RateLimitError as e:
            logger.error(f"Anthropic rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Anthropic connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamDelta]:
        """
        Generate a streaming response from Anthropic.
        
        THEORY: Anthropic Streaming Events
        -----------------------------------
        Anthropic uses a more complex event structure:
        
        1. message_start: Contains input_tokens count
        2. content_block_start: Start of a content block
        3. content_block_delta: Actual text content
        4. content_block_stop: End of content block
        5. message_delta: Contains output_tokens and stop_reason
        6. message_stop: Stream complete
        
        Example events:
            {"type":"message_start","message":{"usage":{"input_tokens":25}}}
            {"type":"content_block_delta","delta":{"text":"Hello"}}
            {"type":"message_delta","usage":{"output_tokens":10}}
        
        We normalize this to match OpenAI's simpler format.
        """
        model = model or self._default_model
        max_tokens = max_tokens or 4096
        
        # Extract system message
        system_content, filtered_messages = self._extract_system_message(messages)
        
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=filtered_messages,  # type: ignore
                system=system_content or "",
                temperature=temperature,
            ) as stream:
                async for event in stream:
                    # Handle different event types
                    if event.type == "message_start":
                        # Get input token count
                        if hasattr(event.message, "usage"):
                            prompt_tokens = event.message.usage.input_tokens
                    
                    elif event.type == "content_block_delta":
                        # Actual content
                        if hasattr(event.delta, "text"):
                            yield StreamDelta(content=event.delta.text)
                    
                    elif event.type == "message_delta":
                        # Final stats
                        if hasattr(event, "usage"):
                            completion_tokens = event.usage.output_tokens
                        
                        yield StreamDelta(
                            content="",
                            finish_reason=event.delta.stop_reason or "stop",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                
        except RateLimitError as e:
            logger.error(f"Anthropic rate limit exceeded during stream: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Anthropic connection error during stream: {e}")
            raise
        except APIError as e:
            logger.error(f"Anthropic API error during stream: {e}")
            raise
