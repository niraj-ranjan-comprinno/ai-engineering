# Week 1: Theoretical Deep Dive

This document explains the theoretical concepts behind the AI Gateway in depth. Read this alongside the code to understand not just *how* but *why*.

**Current Setup:**
- Provider: AWS Bedrock
- Model: Claude Sonnet 4.6
- Authentication: AWS CLI Profile

## Table of Contents

1. [Python for AI Engineering](#1-python-for-ai-engineering)
2. [Async Python Deep Dive](#2-async-python-deep-dive)
3. [HTTP & Streaming Protocols](#3-http--streaming-protocols)
4. [Transformer Tokenization](#4-transformer-tokenization)
5. [LLM API Architecture](#5-llm-api-architecture)
6. [AWS Bedrock Specifics](#6-aws-bedrock-specifics)
7. [Cost Economics](#7-cost-economics)
8. [Design Patterns](#8-design-patterns)

---

## 1. Python for AI Engineering

### Why Python?

Python dominates AI/ML for several reasons:

1. **Rich ecosystem**: NumPy, PyTorch, Transformers, LangChain
2. **Fast iteration**: Dynamic typing, REPL-driven development
3. **Glue language**: Easy to integrate C/Rust for performance
4. **Community**: Vast majority of AI research uses Python

### Key Language Features Used

#### Type Hints (PEP 484)

```python
def chat(
    messages: list[dict],           # Parameter types
    model: str | None = None,       # Optional with default
    temperature: float = 0.7,       # Numeric constraints
) -> LLMResponse:                   # Return type
    ...
```

**Why type hints matter for AI code:**
- Catch bugs before runtime (with mypy)
- Self-documenting code
- IDE autocomplete and error detection
- Pydantic uses them for validation

#### Abstract Base Classes

```python
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict]) -> LLMResponse:
        pass
```

**Purpose:**
- Define contracts for implementations
- Python raises TypeError if methods aren't implemented
- Enable polymorphism (treat different providers the same)

---

## 2. Async Python Deep Dive

### The Problem: I/O Bound Operations

LLM API calls are **I/O bound** - most time is spent waiting:

```
Request → [Network: 50ms] → [LLM Processing: 2000ms] → [Network: 50ms] → Response
          ↑                                                            ↑
          Waiting...                                               Waiting...
```

During this waiting, a synchronous server does NOTHING. An async server can process other requests.

### Event Loop Basics

```
┌─────────────────────────────────────────────────────────────────┐
│                         Event Loop                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Request 1│  │ Request 2│  │ Request 3│  │ Request 4│        │
│  │  waiting │  │ running  │  │  waiting │  │  ready   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│  The loop checks: "Who is ready to run? Who is waiting?"         │
│  Switches between tasks at await points (no OS threads!)         │
└─────────────────────────────────────────────────────────────────┘
```

### async/await Mechanics

```python
async def get_completion(prompt: str) -> str:
    """
    'async def' creates a coroutine - a pausable function.
    """
    # 'await' says: "This might take time. Let others run."
    response = await client.chat(prompt)
    
    # Code after await runs when the result is ready
    return response.content
```

**Key insight**: `await` doesn't block the thread - it yields control back to the event loop.

### Async Generators for Streaming

```python
async def stream_tokens() -> AsyncIterator[str]:
    """
    Async generator: yields values as they become available.
    """
    async for chunk in api_stream:
        yield chunk.content  # Pause, yield, resume

# Consumption
async for token in stream_tokens():
    print(token, end="")
```

### boto3 and Async

boto3 is **synchronous**, but we use it in async code via thread pool:

```python
async def chat(self, messages):
    # Run sync boto3 in thread pool - doesn't block event loop
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,  # Default thread pool
        lambda: self._client.invoke_model(...)
    )
    return response
```

---

## 3. HTTP & Streaming Protocols

### Traditional HTTP Request/Response

```
Client                          Server
  │                                │
  │─────── POST /chat ────────────>│
  │                                │
  │                         [Processing 3s]
  │                                │
  │<────── 200 OK ─────────────────│
  │        {"content": "..."}      │
```

**Problem**: Client waits 3 seconds seeing nothing.

### Server-Sent Events (SSE)

```
Client                          Server
  │                                │
  │─────── POST /chat ────────────>│
  │        Accept: text/event-stream
  │                                │
  │<────── 200 OK ─────────────────│
  │        Content-Type: text/event-stream
  │                                │
  │<── event: chunk ───────────────│
  │    data: {"delta": "Hello"}    │
  │                                │
  │<── event: chunk ───────────────│
  │    data: {"delta": " World"}   │
  │                                │
  │<── event: done ────────────────│
  │    data: [DONE]                │
  │                                │
  └────────────────────────────────┘
```

**Benefits:**
- First token visible in ~100ms
- Single HTTP connection (efficient)
- Standard protocol (works through proxies)

### SSE Wire Format

```
event: chunk
data: {"delta": "Hello"}

event: chunk
data: {"delta": " World"}

event: complete
data: {"usage": {"total_tokens": 28}}

event: done
data: [DONE]

```

- `event:` - Event type
- `data:` - Payload (JSON in our case)
- Empty line (`\n\n`) marks end of event

---

## 4. Transformer Tokenization

### Why Tokens, Not Words?

**Problem with word-based approaches:**
- Unknown words: "Pneumonoultramicroscopicsilicovolcanoconiosis"
- Morphology: "running", "ran", "runs" are different words
- Languages: Chinese has no spaces
- Code: `function_name` is one word?

### Byte Pair Encoding (BPE)

**Training process:**
1. Start with character vocabulary
2. Count all adjacent pairs in training data
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reached

**Example:**
```
"Hello world" → ["Hello", " world"] → 2 tokens
"Tokenization" → ["Token", "ization"] → 2 tokens
"def fibonacci(n):" → ["def", " fib", "onacci", "(n", "):"] → 5 tokens
```

### Token Costs in Chat

Chat completions add overhead tokens:

```
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

Each message adds ~4 tokens for markers.

---

## 5. LLM API Architecture

### Messages Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "Show me an example"},
]
```

**Roles:**
- `system`: Sets behavior/persona (processed specially by model)
- `user`: Human input
- `assistant`: Model's previous responses (for context)

### Key Parameters

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `temperature` | 0.0 - 2.0 | Controls randomness |
| `max_tokens` | 1 - context_limit | Maximum output length |
| `top_p` | 0.0 - 1.0 | Nucleus sampling |

**Temperature explained:**
- `0.0`: Always pick most likely token (deterministic)
- `0.7`: Good balance of creativity and coherence
- `1.5+`: Very random, potentially nonsensical

### Statelessness

LLMs are **stateless** - they don't remember previous requests!

```
Request 1: "My name is Niraj"
Response 1: "Nice to meet you, Niraj!"

Request 2: "What is my name?"
Response 2: "I don't know your name."  ← No memory!
```

To have a conversation, you must send ALL previous messages with each request.

---

## 6. AWS Bedrock Specifics

### What is AWS Bedrock?

A fully managed service providing access to foundation models:
- Anthropic Claude
- Meta Llama
- Amazon Titan
- Mistral
- And more...

### Authentication

Bedrock uses AWS credentials, NOT API keys:

```
AWS_PROFILE=ai-learning
      │
      ▼
~/.aws/credentials
[ai-learning]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
      │
      ▼
boto3.Session(profile_name='ai-learning')
      │
      ▼
AWS Signature V4 signed request
```

### Model IDs vs Inference Profiles

**Old way (doesn't work for newer models):**
```python
model_id = "anthropic.claude-sonnet-4-6"  # ❌ Raw model ID
```

**New way (required for newer models):**
```python
model_id = "us.anthropic.claude-sonnet-4-6"  # ✅ Inference profile
```

**Why inference profiles?**
- Managed throughput
- Cross-region routing
- Automatic failover
- Simplified billing

### Bedrock Request Format (Claude)

```python
{
    "anthropic_version": "bedrock-2023-05-31",  # Required!
    "max_tokens": 4096,
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "system": "You are helpful"  # System prompt is separate!
}
```

**Note:** Unlike OpenAI, Bedrock/Anthropic has `system` as a separate field, not in messages array.

### Bedrock Response Format

```python
{
    "content": [
        {"type": "text", "text": "Hello! How can I help?"}
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 8
    },
    "stop_reason": "end_turn"  # Not "stop" like OpenAI!
}
```

---

## 7. Cost Economics

### Understanding Token Pricing

**Pricing structure:**
- Input tokens: What you send
- Output tokens: What you get back (usually 2-4x more expensive!)

**Why output costs more:**
- Input: Single forward pass through model
- Output: One forward pass PER TOKEN generated

### Cost Calculation

```python
def calculate_cost(prompt_tokens, completion_tokens, model):
    input_price, output_price = MODEL_PRICING[model]  # Per 1M tokens
    
    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    
    return input_cost + output_cost
```

### Cost Optimization Strategies

**1. Use smaller models when possible**
```
Task: "Extract email from text"
Claude Sonnet: Works, but overkill
Claude Haiku: Just as good, 10x cheaper
```

**2. Keep system prompts concise**
```
Long: 500 tokens × 1000 requests = 500K tokens
Short: 50 tokens × 1000 requests = 50K tokens
Savings: 90%!
```

**3. Limit max_tokens**
```python
# For yes/no questions
max_tokens=10  # Don't generate essays for simple questions
```

**4. Cache common responses**
```python
cache = {}
if prompt in cache:
    return cache[prompt]  # Free!
```

---

## 8. Design Patterns

### Strategy Pattern (Providers)

```python
# Define strategy interface
class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages): pass

# Implement strategies
class BedrockProvider(BaseLLMProvider):
    async def chat(self, messages):
        return await self._call_bedrock(messages)

class OpenAIProvider(BaseLLMProvider):
    async def chat(self, messages):
        return await self._call_openai(messages)

# Use strategy
provider = get_provider("bedrock")  # or "openai"
response = await provider.chat(messages)  # Same interface!
```

### Factory Pattern (Router)

```python
class LLMRouter:
    def __init__(self):
        self._providers = {}
    
    def register(self, name: str, provider: BaseLLMProvider):
        self._providers[name] = provider
    
    def get_provider(self, name: str) -> BaseLLMProvider:
        return self._providers[name]
```

### Dependency Injection (FastAPI)

```python
# Define dependency
def get_llm_router(settings: Settings = Depends(get_settings)):
    return LLMRouter(...)

# Inject into route
@app.post("/chat")
async def chat(
    request: ChatRequest,
    router: LLMRouter = Depends(get_llm_router)  # Injected!
):
    return await router.chat(request.messages)
```

**Benefits:**
- Testable (inject mocks)
- Configurable (different dependencies per environment)
- Decoupled (routes don't know how router is created)

### Middleware Pattern

```python
class RequestLoggingMiddleware:
    async def dispatch(self, request, call_next):
        # Before request
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Process request
        response = await call_next(request)
        
        # After request
        duration = time.time() - start_time
        logger.info(f"Request {request_id} took {duration}s")
        
        return response
```

---

## Summary

### Key Concepts Learned

| Concept | Why It Matters |
|---------|----------------|
| Async Python | Handle many concurrent LLM requests |
| SSE Streaming | Better UX with real-time responses |
| Tokenization | Understand costs and limits |
| Provider Abstraction | Swap providers without code changes |
| AWS Bedrock | Enterprise-grade LLM access |
| Cost Tracking | Build sustainable AI products |

### Mental Model

```
User Request
     │
     ▼
┌─────────────────┐
│   AI Gateway    │  ← Validates, routes, tracks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AWS Bedrock    │  ← Manages auth, scaling, models
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Claude Sonnet   │  ← Actually generates response
│     4.6         │
└─────────────────┘
```

---

*Next: Week 2 - RAG Foundations*
