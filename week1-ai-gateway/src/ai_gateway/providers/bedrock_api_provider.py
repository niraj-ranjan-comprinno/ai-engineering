"""
AWS Bedrock API Key Provider Implementation.

=============================================================================
THEORY: Bedrock API Keys (New Feature!)
=============================================================================

AWS Bedrock now supports API keys, similar to OpenAI/Anthropic!
This is much simpler than using AWS credentials (boto3).

Benefits:
- No AWS CLI setup required
- Simple API key authentication
- Works like OpenAI/Anthropic APIs

The endpoint format:
    https://bedrock-runtime.{region}.amazonaws.com/model/{model-id}/invoke
    
With header:
    x-]api-key: your-bedrock-api-key
"""

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from .base import BaseLLMProvider, LLMResponse, StreamDelta

logger = logging.getLogger(__name__)


class BedrockAPIProvider(BaseLLMProvider):
    """
    AWS Bedrock provider using API keys (simpler than boto3!).
    """
    
    # Model mapping: friendly name → Bedrock model ID
    MODEL_MAPPING = {
        "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "claude-3-5-sonnet-v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-sonnet-4": "anthropic.claude-sonnet-4-20250514-v1:0",
        "llama3-70b": "meta.llama3-70b-instruct-v1:0",
        "llama3-8b": "meta.llama3-8b-instruct-v1:0",
        "mistral-large": "mistral.mistral-large-2402-v1:0",
    }
    
    def __init__(
        self,
        api_key: str,
        region: str = "us-east-1",
        default_model: str = "claude-3-5-sonnet",
    ):
        """
        Initialize Bedrock API provider.
        
        Args:
            api_key: Bedrock API key from AWS Console
            region: AWS region (us-east-1, us-west-2, etc.)
            default_model: Default model to use
        """
        self._api_key = api_key
        self._region = region
        self._default_model = default_model
        self._base_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        
        logger.info(f"Bedrock API provider initialized in {region}")
    
    @property
    def name(self) -> str:
        return "bedrock"
    
    @property
    def default_model(self) -> str:
        return self._default_model
    
    @property
    def available_models(self) -> list[str]:
        return list(self.MODEL_MAPPING.keys())
    
    def _get_model_id(self, model: str) -> str:
        """Convert friendly name to Bedrock model ID."""
        return self.MODEL_MAPPING.get(model, model)
    
    def _is_claude_model(self, model_id: str) -> bool:
        return "anthropic" in model_id or "claude" in model_id.lower()
    
    def _build_claude_body(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Build request body for Claude models."""
        system_content = ""
        filtered_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_content += msg["content"] + "\n"
            else:
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": filtered_messages,
        }
        
        if system_content.strip():
            body["system"] = system_content.strip()
        
        return body
    
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a non-streaming response."""
        model = model or self._default_model
        model_id = self._get_model_id(model)
        max_tokens = max_tokens or 4096
        
        # Build request
        body = self._build_claude_body(messages, temperature, max_tokens)
        
        url = f"{self._base_url}/model/{model_id}/invoke"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                },
            )
            response.raise_for_status()
            data = response.json()
        
        # Parse Claude response
        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop"),
        )
    
    async def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamDelta]:
        """Generate a streaming response."""
        model = model or self._default_model
        model_id = self._get_model_id(model)
        max_tokens = max_tokens or 4096
        
        body = self._build_claude_body(messages, temperature, max_tokens)
        
        url = f"{self._base_url}/model/{model_id}/invoke-with-response-stream"
        
        prompt_tokens = 0
        completion_tokens = 0
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                },
            ) as response:
                response.raise_for_status()
                
                buffer = b""
                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    
                    # Parse Bedrock's event stream format
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.strip():
                            continue
                        
                        try:
                            # Bedrock returns binary event stream
                            # Parse the JSON from the chunk
                            event_data = json.loads(line.decode('utf-8'))
                            
                            if "bytes" in event_data:
                                payload = json.loads(
                                    event_data["bytes"].encode().decode('utf-8')
                                )
                            else:
                                payload = event_data
                            
                            # Handle different event types
                            event_type = payload.get("type", "")
                            
                            if event_type == "content_block_delta":
                                delta_text = payload.get("delta", {}).get("text", "")
                                if delta_text:
                                    yield StreamDelta(content=delta_text)
                            
                            elif event_type == "message_start":
                                usage = payload.get("message", {}).get("usage", {})
                                prompt_tokens = usage.get("input_tokens", 0)
                            
                            elif event_type == "message_delta":
                                usage = payload.get("usage", {})
                                completion_tokens = usage.get("output_tokens", 0)
                                stop_reason = payload.get("delta", {}).get("stop_reason")
                                
                                yield StreamDelta(
                                    content="",
                                    finish_reason=stop_reason,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                )
                        except (json.JSONDecodeError, KeyError):
                            continue
