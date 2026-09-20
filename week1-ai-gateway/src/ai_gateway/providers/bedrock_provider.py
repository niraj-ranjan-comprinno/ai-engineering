"""
AWS Bedrock Provider Implementation.

=============================================================================
THEORY: AWS Bedrock for LLM Access
=============================================================================

What is AWS Bedrock?
--------------------
Amazon Bedrock is a fully managed service that provides access to 
foundation models from multiple providers through a single API:

- Anthropic Claude (claude-3-5-sonnet, claude-3-opus, claude-3-haiku)
- Meta Llama (llama-3-70b, llama-3-8b)
- Amazon Titan
- Mistral
- Cohere

Why use Bedrock?
----------------
1. Single billing: All models billed through AWS
2. Enterprise features: VPC, IAM, CloudWatch integration
3. Data privacy: Your data stays in your AWS account
4. No API key management: Uses AWS credentials

Authentication:
---------------
Bedrock uses AWS credentials (not API keys):
- AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
- Or IAM roles (recommended for EC2/Lambda)
- Or AWS CLI profile

Model IDs:
----------
Bedrock uses specific model IDs:
- anthropic.claude-3-5-sonnet-20240620-v1:0
- anthropic.claude-3-haiku-20240307-v1:0
- meta.llama3-70b-instruct-v1:0
"""

import json
import logging
from typing import AsyncIterator, Optional

from .base import BaseLLMProvider, LLMResponse, StreamDelta

logger = logging.getLogger(__name__)

# Check if boto3 is available
try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. Install with: pip install boto3")


class BedrockProvider(BaseLLMProvider):
    """
    AWS Bedrock provider implementation.
    
    THEORY: Bedrock vs Direct API
    -----------------------------
    Bedrock wraps model APIs with AWS-specific features:
    
    Direct API:
        Client → OpenAI/Anthropic API → Response
    
    Bedrock:
        Client → AWS Bedrock → Model Provider → Response
                    ↓
            IAM, VPC, CloudWatch, etc.
    
    The request/response format is slightly different from direct APIs.
    """
    
    # Bedrock model IDs mapped to friendly names
    MODEL_MAPPING = {
        # Claude Sonnet 4.6 (Bedrock Edition) - using inference profile
        "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
        "claude-4-6": "us.anthropic.claude-sonnet-4-6",
        "claude-sonnet": "us.anthropic.claude-sonnet-4-6",
        # Aliases
        "claude-3-5-sonnet": "us.anthropic.claude-sonnet-4-6",
        "claude-3-haiku": "us.anthropic.claude-sonnet-4-6",
    }
    
    # Pricing per 1M tokens (input, output) - varies by region
    MODEL_PRICING = {
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25, 1.25),
        "claude-3-opus": (15.00, 75.00),
        "llama3-70b": (2.65, 3.50),
        "llama3-8b": (0.30, 0.60),
        "mistral-large": (4.00, 12.00),
        "mistral-7b": (0.15, 0.20),
    }
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        default_model: str = "claude-3-5-sonnet",
        profile_name: Optional[str] = None,
    ):
        """
        Initialize Bedrock provider.
        
        THEORY: AWS Credential Chain
        ----------------------------
        boto3 looks for credentials in this order:
        1. Explicit credentials (access_key, secret_key)
        2. Environment variables (AWS_ACCESS_KEY_ID, etc.)
        3. Shared credentials file (~/.aws/credentials)
        4. AWS config file (~/.aws/config)
        5. IAM role (for EC2/Lambda)
        
        For development, use:
        - Environment variables, or
        - AWS CLI: `aws configure`
        
        For production:
        - IAM roles attached to EC2/Lambda
        
        Args:
            region_name: AWS region (must have Bedrock access)
            default_model: Friendly model name
            profile_name: AWS CLI profile name (optional)
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for Bedrock. "
                "Install with: pip install boto3"
            )
        
        # Configure boto3 client
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
        self._region = region_name
        self._default_model = default_model
        
        logger.info(
            f"Bedrock provider initialized in {region_name} "
            f"with default model: {default_model}"
        )
    
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
    
    def _is_claude_model(self, model: str) -> bool:
        """Check if model is Claude (uses Messages API)."""
        model_id = self._get_model_id(model)
        return "anthropic" in model_id or "claude" in model.lower()
    
    def _is_llama_model(self, model: str) -> bool:
        """Check if model is Llama."""
        model_id = self._get_model_id(model)
        return "meta" in model_id or "llama" in model.lower()
    
    def _build_claude_body(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> dict:
        """
        Build request body for Claude models on Bedrock.
        
        THEORY: Bedrock Claude Format
        -----------------------------
        Bedrock's Claude API is slightly different from Anthropic's direct API:
        
        Direct Anthropic:
            client.messages.create(
                model="claude-3-5-sonnet",
                messages=[...],
                system="..."
            )
        
        Bedrock:
            client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet...",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [...],
                    "system": "..."
                })
            )
        
        TOOL USE FORMAT:
        ----------------
        When tools are provided, Claude can return tool_use blocks:
        {
            "type": "tool_use",
            "id": "toolu_01...",
            "name": "calculator",
            "input": {"expression": "2 + 2"}
        }
        """
        # Extract system message from messages
        system_content = system or ""
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
        
        # Add tools if provided
        if tools:
            body["tools"] = tools
        
        return body
    
    def _build_llama_body(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """
        Build request body for Llama models on Bedrock.
        
        THEORY: Llama Prompt Format
        ---------------------------
        Llama uses a specific prompt format:
        
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are helpful.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Hello<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        Bedrock handles some of this, but we need to format properly.
        """
        # Build prompt from messages
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "user":
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "assistant":
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")
        
        # Add assistant header for response
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        
        return {
            "prompt": "".join(prompt_parts),
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }
    
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Generate a non-streaming response from Bedrock.
        
        Note: boto3 is synchronous, so we run it in a thread pool
        to avoid blocking the event loop.
        
        Args:
            messages: Conversation messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            tools: Optional list of tool schemas for function calling
            system: Optional system prompt
        """
        import asyncio
        
        model = model or self._default_model
        model_id = self._get_model_id(model)
        max_tokens = max_tokens or 4096
        
        # Build request body based on model type
        if self._is_claude_model(model):
            body = self._build_claude_body(messages, temperature, max_tokens, tools, system)
        elif self._is_llama_model(model):
            body = self._build_llama_body(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Run synchronous boto3 call in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if self._is_claude_model(model):
            # Claude returns content as a list of blocks
            # Can be text blocks or tool_use blocks
            content = response_body.get("content", [])
            prompt_tokens = response_body["usage"]["input_tokens"]
            completion_tokens = response_body["usage"]["output_tokens"]
            stop_reason = response_body.get("stop_reason", "end_turn")
            finish_reason = stop_reason
        elif self._is_llama_model(model):
            content = response_body["generation"]
            # Llama doesn't return token counts, estimate
            prompt_tokens = len(str(messages)) // 4  # rough estimate
            completion_tokens = len(content) // 4
            stop_reason = response_body.get("stop_reason", "stop")
            finish_reason = stop_reason
        else:
            raise ValueError(f"Cannot parse response for model: {model}")
        
        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )
    
    async def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamDelta]:
        """
        Generate a streaming response from Bedrock.
        
        THEORY: Bedrock Streaming
        -------------------------
        Bedrock uses invoke_model_with_response_stream for streaming.
        The response is an event stream with chunks.
        
        Note: We need to run the synchronous stream in a way that
        doesn't block the event loop.
        """
        import asyncio
        
        model = model or self._default_model
        model_id = self._get_model_id(model)
        max_tokens = max_tokens or 4096
        
        # Build request body
        if self._is_claude_model(model):
            body = self._build_claude_body(messages, temperature, max_tokens)
        elif self._is_llama_model(model):
            body = self._build_llama_body(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Get streaming response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        )
        
        # Process stream
        prompt_tokens = 0
        completion_tokens = 0
        
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            
            if self._is_claude_model(model):
                if chunk.get("type") == "content_block_delta":
                    yield StreamDelta(content=chunk["delta"].get("text", ""))
                elif chunk.get("type") == "message_delta":
                    completion_tokens = chunk.get("usage", {}).get("output_tokens", 0)
                    yield StreamDelta(
                        content="",
                        finish_reason=chunk.get("delta", {}).get("stop_reason"),
                        completion_tokens=completion_tokens,
                    )
                elif chunk.get("type") == "message_start":
                    prompt_tokens = chunk.get("message", {}).get("usage", {}).get("input_tokens", 0)
                    
            elif self._is_llama_model(model):
                if "generation" in chunk:
                    yield StreamDelta(content=chunk["generation"])
                if chunk.get("stop_reason"):
                    yield StreamDelta(
                        content="",
                        finish_reason=chunk["stop_reason"],
                    )
