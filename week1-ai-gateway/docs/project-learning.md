# Week 1: AI Gateway - Learning Guide

## For Software Engineers Learning AI Engineering

---

## 📌 What is This Project?

This project is an **AI Gateway** - a backend service that acts as a unified interface between your applications and Large Language Model (LLM) providers like AWS Bedrock (Claude), OpenAI (GPT), and Anthropic.

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Your Web App   │      │                 │      │   AWS Bedrock   │
├─────────────────┤      │                 │      │   (Claude)      │
│  Mobile App     │ ───> │   AI Gateway    │ ───> ├─────────────────┤
├─────────────────┤      │   (This Project)│      │   OpenAI        │
│  Internal Tool  │      │                 │      │   (GPT-4)       │
├─────────────────┤      │                 │      ├─────────────────┤
│  CLI Script     │      │                 │      │   Anthropic     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

**Think of it as:** A smart proxy that handles all LLM communication for your organization.

---

## 🎯 Problem Statement

### The Challenge

Your company wants to build AI-powered features across multiple products. Without a gateway, each team faces these problems:

| Problem | Impact |
|---------|--------|
| **API Key Sprawl** | Every app has hardcoded keys → Security risk |
| **No Cost Visibility** | No idea how much each team/feature costs → Budget overruns |
| **Provider Lock-in** | Direct OpenAI integration → Hard to switch providers |
| **No Fallback** | OpenAI down = Your app down → Reliability issues |
| **Duplicate Code** | Every team writes token counting, error handling → Wasted effort |
| **No Monitoring** | Can't track usage, latency, errors → Blind spots |

### The Solution

Build a centralized AI Gateway that:

1. ✅ Manages API keys securely (one place)
2. ✅ Tracks token usage and costs (per request)
3. ✅ Supports multiple providers (Bedrock, OpenAI, Anthropic)
4. ✅ Provides automatic fallback (if one provider fails)
5. ✅ Offers unified API (all apps use same interface)
6. ✅ Logs everything (debugging, monitoring, billing)

---

## 🚀 How to Run This Project

### Prerequisites

```bash
# Check Python version (need 3.11+)
python3 --version

# Check AWS CLI is configured
aws sts get-caller-identity --profile ai-learning
```

### Step 1: Navigate to Project

```bash
cd /Users/nirajranjan/Documents/developer/learning/week1-ai-gateway
```

### Step 2: Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Step 3: Start the Server

```bash
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload --port 8000
```

You should see:
```
INFO:     AI Gateway starting up...
INFO:     Default provider: bedrock
INFO:     Default model: claude-sonnet-4-6
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 4: Test It Works

```bash
# Health check
curl http://localhost:8000/v1/health

# Make a chat request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": false}'
```

---

## 📚 What Knowledge You Will Gain

After completing this project, you will understand:

| Skill Area | What You'll Learn |
|------------|-------------------|
| **Python Async** | `async/await`, event loops, concurrent programming |
| **FastAPI** | Building REST APIs, dependency injection, middleware |
| **Pydantic** | Data validation, settings management, type hints |
| **LLM Fundamentals** | Tokens, temperature, streaming, context windows |
| **AWS Bedrock** | Inference profiles, boto3, authentication |
| **Design Patterns** | Provider abstraction, router pattern, middleware chain |
| **Cost Management** | Token pricing, cost calculation, optimization |
| **Production Skills** | Logging, metrics, error handling, fallback strategies |

---

## 📖 Detailed Learning Topics

---

# Topic 1: Understanding LLMs and Tokens

## What is an LLM?

A **Large Language Model (LLM)** is a neural network trained on massive amounts of text data. It predicts the next word (token) based on the input it receives.

**Key Characteristics:**
- Input: Text (called "prompt")
- Output: Text (called "completion")
- Stateless: Doesn't remember previous requests
- Probabilistic: Same input can give different outputs

## What are Tokens?

LLMs don't see words - they see **tokens**. A token is a piece of text, roughly 4 characters on average.

**Examples:**

| Text | Tokens | Count |
|------|--------|-------|
| "Hello" | ["Hello"] | 1 |
| "Hello, world!" | ["Hello", ",", " world", "!"] | 4 |
| "AI engineering" | ["AI", " engineering"] | 2 |
| "नमस्ते" (Hindi) | ["न", "म", "स", "्", "त", "े"] | 6 |

**Why tokens matter:**
1. **Billing**: You pay per token (input + output)
2. **Limits**: Context window is measured in tokens
3. **Cost**: Non-English text often uses more tokens

### 💻 Hands-On Exercise 1.1: Count Tokens

```bash
# Count tokens in a message
curl -X POST http://localhost:8000/v1/tokens/count \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is Python?"}]}'
```

**Try these and compare token counts:**
- "Hi" vs "Hello, how are you today?"
- English text vs Hindi text
- Plain text vs code snippet

### 📁 Code to Study

**File:** `src/ai_gateway/tokenizer.py`

Key functions:
- `count_tokens()` - Count tokens in text
- `count_message_tokens()` - Count tokens in chat messages
- `calculate_cost()` - Calculate USD cost

---

# Topic 2: The Chat Completion API Pattern

## Message Roles

LLMs use a role-based message format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},  # Instructions
    {"role": "user", "content": "What is Python?"},                 # User's question
    {"role": "assistant", "content": "Python is..."},               # AI's previous response
    {"role": "user", "content": "Show me an example"}               # Follow-up
]
```

| Role | Purpose | When to Use |
|------|---------|-------------|
| `system` | Set behavior/personality | Once at start |
| `user` | Human's messages | Each user input |
| `assistant` | AI's previous responses | For conversation history |

### 💻 Hands-On Exercise 2.1: System Prompts

```bash
# Without system prompt
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain variables"}],
    "stream": false
  }'

# With system prompt - pirate personality
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a pirate. Speak like a pirate in all responses."},
      {"role": "user", "content": "Explain variables"}
    ],
    "stream": false
  }'
```

**Observe:** Same question, different personality!

### 💻 Hands-On Exercise 2.2: Conversation History

```bash
# Multi-turn conversation
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Niraj"},
      {"role": "assistant", "content": "Nice to meet you, Niraj!"},
      {"role": "user", "content": "What is my name?"}
    ],
    "stream": false
  }'
```

**Key insight:** The LLM only knows what's in the messages array. It has no memory between requests!

### 📁 Code to Study

**File:** `src/ai_gateway/models.py`

Look at:
- `Message` class - Validates role and content
- `ChatRequest` class - Full request structure

---

# Topic 3: Temperature and Creativity

## What is Temperature?

Temperature controls **randomness** in the model's output.

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Always picks most likely token | Math, code, factual answers |
| 0.3 - 0.7 | Balanced (default range) | General assistant |
| 1.0+ | More random, creative | Creative writing, brainstorming |

**Visualization:**

```
Temperature 0:    [Most Likely] ████████████████████ 100%
                  [2nd choice]  
                  [3rd choice]  

Temperature 1:    [Most Likely] ████████████ 50%
                  [2nd choice]  ██████ 30%
                  [3rd choice]  ████ 20%
```

### 💻 Hands-On Exercise 3.1: Temperature Comparison

```bash
# Temperature 0 - Run 3 times, same result!
for i in 1 2 3; do
  echo "=== Run $i ==="
  curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Complete: The sky is"}], "temperature": 0, "stream": false}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['content'])"
done

# Temperature 1 - Run 3 times, different results!
for i in 1 2 3; do
  echo "=== Run $i ==="
  curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Complete: The sky is"}], "temperature": 1.0, "stream": false}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['content'])"
done
```

### 📁 Code to Study

**File:** `src/ai_gateway/models.py`

Look at:
- `ChatRequest.temperature` - Field validation with `ge=0.0, le=2.0`

---

# Topic 4: Streaming vs Non-Streaming

## The Difference

**Non-Streaming:**
```
User sends request → Waits 3 seconds → Gets complete response
```

**Streaming:**
```
User sends request → Gets "Hello" → Gets " World" → Gets "!" → Done
                     (instant)      (100ms)        (200ms)
```

Streaming feels **much faster** because users see progress immediately.

## How Streaming Works (Server-Sent Events)

```
HTTP Response keeps connection open:

event: chunk
data: {"delta": "Hello"}

event: chunk  
data: {"delta": " World"}

event: complete
data: {"usage": {"total_tokens": 10}}

event: done
data: [DONE]
```

### 💻 Hands-On Exercise 4.1: Compare Streaming

```bash
# Non-streaming - notice the wait
echo "=== Non-Streaming ==="
time curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a short poem about coding"}], "stream": false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['content'])"

# Streaming - words appear one by one
echo ""
echo "=== Streaming ==="
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a short poem about coding"}], "stream": true}'
```

### 📁 Code to Study

**File:** `src/ai_gateway/api/routes.py`

Look at:
- `stream_chat_completions()` - Async generator for SSE
- `EventSourceResponse` - Sends SSE to client

---

# Topic 5: Async/Await in Python

## Why Async for LLM Apps?

LLM API calls take 1-30 seconds. Without async, your server blocks:

```python
# Synchronous - BLOCKS everything
def handle_request():
    response = call_llm()  # 5 seconds - nothing else can happen!
    return response

# 10 users at same time:
# User 1: 0-5 sec
# User 2: 5-10 sec (waiting!)
# User 10: 45-50 sec (waiting 45 seconds!)
```

With async:

```python
# Asynchronous - yields control while waiting
async def handle_request():
    response = await call_llm()  # Yields, serves other users!
    return response

# 10 users at same time:
# All 10 start immediately
# All 10 complete in ~5-6 seconds!
```

## Key Syntax

```python
# 1. Define an async function
async def my_function():
    pass

# 2. Await - pause and let other tasks run
result = await some_async_call()

# 3. Run multiple tasks concurrently
results = await asyncio.gather(task1(), task2(), task3())

# 4. Iterate over async stream
async for chunk in stream:
    yield chunk
```

### 💻 Hands-On Exercise 5.1: Understand Async

Create a file `test_async.py`:

```python
import asyncio
import time

async def slow_api_call(user_id):
    print(f"User {user_id}: Starting...")
    await asyncio.sleep(2)  # Simulate API call
    print(f"User {user_id}: Done!")
    return f"Result for user {user_id}"

async def main():
    start = time.time()
    
    # Run 3 "API calls" concurrently
    results = await asyncio.gather(
        slow_api_call(1),
        slow_api_call(2),
        slow_api_call(3),
    )
    
    print(f"\nTotal time: {time.time() - start:.1f}s")
    print(f"Results: {results}")

asyncio.run(main())
```

Run it:
```bash
python3 test_async.py
```

**Expected:** All 3 complete in ~2 seconds, not 6 seconds!

### 📁 Code to Study

**File:** `src/ai_gateway/providers/bedrock_provider.py`

Look at:
- `async def chat()` - Async method definition
- `await loop.run_in_executor()` - Running sync boto3 in thread pool

---

# Topic 6: Pydantic for Data Validation

## What is Pydantic?

Pydantic validates data automatically using Python type hints.

```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    messages: list[Message]               # Must be list
    temperature: float = Field(ge=0, le=2)  # Must be 0-2
    max_tokens: int = Field(default=1024, ge=1)  # Must be >= 1
```

**What happens with invalid data:**

```python
# This raises ValidationError automatically!
request = ChatRequest(messages=[], temperature=5.0)
# Error: temperature must be <= 2, messages must have at least 1 item
```

### 💻 Hands-On Exercise 6.1: Test Validation

```bash
# Invalid: temperature too high
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}], "temperature": 5.0, "stream": false}'

# Invalid: empty messages
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [], "stream": false}'

# Invalid: wrong role
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "superuser", "content": "Hi"}], "stream": false}'
```

**Observe:** Pydantic returns clear error messages!

### 📁 Code to Study

**File:** `src/ai_gateway/models.py`

Look at:
- `Field()` - Adds validation rules
- `Literal["system", "user", "assistant"]` - Restricts to specific values

---

# Topic 7: @lru_cache for Caching

## What is @lru_cache?

`@lru_cache` memoizes (caches) function results. Same input → return cached result.

```python
from functools import lru_cache

@lru_cache
def get_settings():
    print("Loading settings...")  # Only prints ONCE
    return Settings()

# First call - executes function
settings1 = get_settings()  # Prints: Loading settings...

# Subsequent calls - returns cached result
settings2 = get_settings()  # No print! Returns cached object
settings3 = get_settings()  # No print! Same object
```

## Why Use It?

- **Settings**: Load `.env` file once, reuse everywhere
- **Tokenizer**: Load encoding once (~100ms), reuse
- **Simple Singleton**: One instance shared across app

### 📁 Code to Study

**File:** `src/ai_gateway/config.py`

Look at:
- `@lru_cache` decorator on `get_settings()`
- Why this creates a singleton pattern

**File:** `src/ai_gateway/tokenizer.py`

Look at:
- `@lru_cache(maxsize=10)` on `get_encoding()`
- Caches up to 10 different encodings

---

# Topic 8: Provider Abstraction Pattern

## The Problem

Different LLM providers have different APIs:

```python
# OpenAI
response = openai.chat.completions.create(model="gpt-4", messages=[...])
content = response.choices[0].message.content

# Anthropic
response = anthropic.messages.create(model="claude-3", messages=[...])
content = response.content[0].text

# Bedrock
response = bedrock.invoke_model(modelId="...", body=json.dumps({...}))
content = json.loads(response['body'].read())["content"][0]["text"]
```

## The Solution: Abstract Base Class

Define an interface that ALL providers must implement:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages, model, ...) -> LLMResponse:
        pass
    
    @abstractmethod
    async def stream_chat(self, messages, ...) -> AsyncIterator[StreamDelta]:
        pass
```

Each provider implements it differently, but the **interface is the same**:

```python
# Your code doesn't care which provider!
provider = get_provider()  # Could be OpenAI, Bedrock, anything
response = await provider.chat(messages)  # Same method for all
```

### 📁 Code to Study

**File:** `src/ai_gateway/providers/base.py`

Look at:
- `BaseLLMProvider` - Abstract base class
- `LLMResponse` - Unified response format
- `StreamDelta` - Unified streaming chunk

**File:** `src/ai_gateway/providers/bedrock_provider.py`

Look at:
- How it implements the base interface
- Model mapping (friendly names → Bedrock IDs)

---

# Topic 9: Router Pattern for Fallback

## What is the Router?

The Router selects which provider to use and handles fallback:

```
Request arrives
     │
     ▼
┌─────────────────────────────────┐
│  Router: Try Bedrock first      │
│  └─> Bedrock fails? Try OpenAI  │
│      └─> OpenAI fails? Error    │
└─────────────────────────────────┘
```

## Why It Matters

| Scenario | Without Router | With Router |
|----------|----------------|-------------|
| Provider down | Your app is down | Auto-fallback |
| Rate limited | 429 errors | Try another provider |
| Cost optimization | Always same model | Route by complexity |

### 💻 Hands-On Exercise 9.1: Check Providers

```bash
# See available providers
curl http://localhost:8000/v1/providers | python3 -m json.tool
```

### 📁 Code to Study

**File:** `src/ai_gateway/providers/router.py`

Look at:
- `LLMRouter` class - Central routing logic
- `chat()` method - Fallback implementation
- How providers are registered

---

# Topic 10: Middleware for Cross-Cutting Concerns

## What is Middleware?

Middleware wraps every request/response to add common functionality:

```
Request → [Logging] → [Auth] → [Rate Limit] → Your Route → Response
              ↑                                    │
              └────────────────────────────────────┘
                      (Also logs response)
```

## What Our Middleware Does

1. **Request ID**: Assigns unique ID for tracing
2. **Timing**: Measures request latency
3. **Logging**: Logs start/end of each request
4. **Metrics**: Tracks counts, errors, latency

### 💻 Hands-On Exercise 10.1: See Metrics

```bash
# Make a few requests first
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": false}' > /dev/null

# Check metrics
curl http://localhost:8000/metrics | python3 -m json.tool
```

### 💻 Hands-On Exercise 10.2: Request ID Tracing

```bash
# Notice the X-Request-ID in response headers
curl -i -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": false}' 2>&1 | grep -i request-id
```

### 📁 Code to Study

**File:** `src/ai_gateway/api/middleware.py`

Look at:
- `RequestLoggingMiddleware` - How it wraps requests
- `MetricsStore` - Simple in-memory metrics
- `request_id` generation and propagation

---

# Topic 11: Cost Management

## How LLM Pricing Works

```
Cost = (Input Tokens × Input Price) + (Output Tokens × Output Price)
```

Prices are per 1 million tokens:

| Model | Input (per 1M) | Output (per 1M) |
|-------|----------------|-----------------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |
| claude-3.5-sonnet | $3.00 | $15.00 |
| gpt-4 | $30.00 | $60.00 |

**Key insight:** Output tokens cost 2-5x more than input!

### 💻 Hands-On Exercise 11.1: Compare Costs

```bash
# Short response
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2? One word answer."}], "stream": false}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Tokens: {d[\"usage\"][\"total_tokens\"]}, Cost: \${d[\"cost\"][\"total_cost\"]}')"

# Long response
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain Python in detail with examples"}], "stream": false}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Tokens: {d[\"usage\"][\"total_tokens\"]}, Cost: \${d[\"cost\"][\"total_cost\"]}')"
```

### 📁 Code to Study

**File:** `src/ai_gateway/tokenizer.py`

Look at:
- `MODEL_PRICING` dictionary
- `calculate_cost()` function

---

## 🎯 Project Structure Reference

```
week1-ai-gateway/
├── src/ai_gateway/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings (@lru_cache, Pydantic)
│   ├── models.py            # Request/Response schemas (Pydantic)
│   ├── tokenizer.py         # Token counting & cost calculation
│   ├── api/
│   │   ├── routes.py        # HTTP endpoints (streaming, non-streaming)
│   │   └── middleware.py    # Logging, metrics, request ID
│   └── providers/
│       ├── base.py          # Abstract interface
│       ├── bedrock_provider.py  # AWS Bedrock implementation
│       ├── openai_provider.py   # OpenAI implementation
│       └── router.py        # Provider selection & fallback
├── docs/                    # Documentation
├── .env                     # Configuration
└── pyproject.toml          # Dependencies
```

---

## ✅ Learning Checklist

Use this to track your progress:

- [ ] Made first API call to the gateway
- [ ] Understand tokens vs words
- [ ] Tested system prompts and conversation history
- [ ] Compared temperature 0 vs temperature 1
- [ ] Saw streaming vs non-streaming difference
- [ ] Understand async/await basics
- [ ] Tested Pydantic validation errors
- [ ] Understand @lru_cache caching
- [ ] Read the provider abstraction code
- [ ] Checked the router fallback logic
- [ ] Viewed metrics endpoint
- [ ] Understand cost calculation

---

## 🔗 Quick Reference

### Start Server
```bash
cd week1-ai-gateway
source .venv/bin/activate
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/providers` | GET | List available providers |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/tokens/count` | POST | Count tokens |
| `/metrics` | GET | View metrics |

### Key Environment Variables
| Variable | Description |
|----------|-------------|
| `AWS_PROFILE` | AWS credentials profile |
| `USE_BEDROCK` | Enable Bedrock provider |
| `DEFAULT_MODEL` | Default model to use |

---

## 📚 Next Steps

After completing this project:

1. **Week 2: RAG Foundations** - Build a knowledge retrieval system
2. **Week 3: Agent Basics** - Create an AI agent with tool use

---

*Happy Learning! 🚀*
