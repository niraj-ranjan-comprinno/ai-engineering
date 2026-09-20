# AI Gateway Architecture

This document describes the architecture of the AI Gateway, explaining how components interact and why design decisions were made.

**Current Configuration:**
- Provider: AWS Bedrock
- Model: Claude Sonnet 4.6 (`us.anthropic.claude-sonnet-4-6`)
- Authentication: AWS CLI Profile (`ai-learning`)

## System Overview

```
                                    ┌─────────────────────────────────────┐
                                    │           AI Gateway                 │
                                    │                                      │
┌──────────────┐                    │  ┌────────────────────────────────┐ │
│    Client    │                    │  │        FastAPI App              │ │
│  (Browser,   │ ──── HTTP ────────>│  │                                │ │
│   CLI, App)  │                    │  │  ┌─────────┐    ┌───────────┐ │ │
└──────────────┘                    │  │  │Middleware│───>│  Routes   │ │ │
                                    │  │  │(Logging) │    │           │ │ │
                                    │  │  └─────────┘    └─────┬─────┘ │ │
                                    │  └───────────────────────│───────┘ │
                                    │                          │         │
                                    │  ┌───────────────────────▼───────┐ │
                                    │  │         LLM Router            │ │
                                    │  │                               │ │
                                    │  │  ┌─────────────────────────┐ │ │
                                    │  │  │   Bedrock Provider      │ │ │
                                    │  │  │   (Claude Sonnet 4.6)   │ │ │
                                    │  │  └───────────┬─────────────┘ │ │
                                    │  └──────────────│───────────────┘ │
                                    └─────────────────│─────────────────┘
                                                      │
                                                      ▼
                                         ┌─────────────────────────┐
                                         │      AWS Bedrock        │
                                         │   (us-east-1 region)    │
                                         │                         │
                                         │  us.anthropic.claude-   │
                                         │  sonnet-4-6             │
                                         └─────────────────────────┘
```

## Component Responsibilities

### 1. FastAPI Application (`main.py`)

**Purpose**: Application entry point and lifecycle management

**Responsibilities**:
- Initialize the ASGI application
- Configure middleware stack
- Mount routers
- Handle application startup/shutdown
- Global error handling

**Key code:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("AI Gateway starting up...")
    yield
    # Shutdown
    logger.info("AI Gateway shutting down...")

app = FastAPI(title="AI Gateway", lifespan=lifespan)
```

### 2. Configuration (`config.py`)

**Purpose**: Load and validate settings from environment

**Current settings:**
```python
class Settings(BaseSettings):
    use_bedrock: bool = True
    aws_region: str = "us-east-1"
    aws_profile: str = "ai-learning"
    default_provider: str = "bedrock"
    default_model: str = "claude-sonnet-4-6"
```

**How it works:**
1. Reads `.env` file
2. Overrides with environment variables
3. Validates types with Pydantic
4. Cached with `@lru_cache`

### 3. API Routes (`api/routes.py`)

**Purpose**: HTTP endpoint definitions

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/health` | GET | Health check |
| `/v1/providers` | GET | List available providers |
| `/v1/chat/completions` | POST | Chat completion (streaming/non-streaming) |
| `/v1/tokens/count` | POST | Token counting utility |
| `/metrics` | GET | Application metrics |

### 4. Middleware (`api/middleware.py`)

**Purpose**: Cross-cutting concerns

**Components:**

#### RequestLoggingMiddleware
- Assigns unique request ID
- Logs request start/end
- Records latency
- Adds `X-Request-ID` to response headers

#### MetricsStore
- In-memory metrics aggregation
- Tracks counts, latencies, errors

### 5. Provider System (`providers/`)

**Purpose**: Abstraction over LLM APIs

#### Base Provider (`base.py`)
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages, model, temperature, max_tokens) -> LLMResponse:
        pass
    
    @abstractmethod
    async def stream_chat(self, ...) -> AsyncIterator[StreamDelta]:
        pass
```

#### Bedrock Provider (`bedrock_provider.py`)
```python
class BedrockProvider(BaseLLMProvider):
    MODEL_MAPPING = {
        "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    }
    
    def __init__(self, region_name, default_model, profile_name):
        session = boto3.Session(profile_name=profile_name)
        self._client = session.client('bedrock-runtime', region_name=region_name)
```

#### Router (`router.py`)
- Registers available providers
- Routes requests to correct provider
- Handles fallback on failure

### 6. Data Models (`models.py`)

**Request Models:**
- `Message`: Single chat message (role + content)
- `ChatRequest`: Full request with messages, model, temperature, etc.

**Response Models:**
- `ChatResponse`: Complete response with content, usage, cost
- `StreamChunk`: Single chunk in streaming response
- `StreamComplete`: Final stats after streaming

**Usage Models:**
- `TokenUsage`: prompt_tokens, completion_tokens, total_tokens
- `CostBreakdown`: input_cost, output_cost, total_cost

### 7. Tokenizer (`tokenizer.py`)

**Purpose**: Token counting and cost calculation

**Functions:**
- `count_tokens(text, model)`: Count tokens using tiktoken
- `count_message_tokens(messages, model)`: Count with chat overhead
- `calculate_cost(prompt_tokens, completion_tokens, model)`: USD cost

## Data Flow

### Non-Streaming Request

```
1. Client POST /v1/chat/completions
         │
2. Middleware: Assign request_id, start timer
         │
3. Pydantic: Validate ChatRequest
         │
4. Route handler: Get LLMRouter via Depends()
         │
5. Router: Get BedrockProvider
         │
6. BedrockProvider.chat():
   - Build request body (anthropic_version, messages, etc.)
   - Call boto3 invoke_model() in thread pool
   - Parse response
         │
7. Calculate tokens & cost
         │
8. Build ChatResponse
         │
9. Middleware: Log completion, record metrics
         │
10. Return JSON
```

### Streaming Request

```
1. Client POST /v1/chat/completions (stream=true)
         │
2. Middleware: Assign request_id
         │
3. Create EventSourceResponse with async generator
         │
4. Return response (connection stays open)
         │
5. Generator yields events:
   │
   ├─> BedrockProvider.stream_chat()
   │   - Call invoke_model_with_response_stream()
   │   - Parse event stream
   │
   ├─> For each chunk:
   │   yield {"event": "chunk", "data": {"delta": "..."}}
   │
   ├─> When complete:
   │   yield {"event": "complete", "data": {"usage": {...}}}
   │
   └─> yield {"event": "done", "data": "[DONE]"}
   
6. Connection closes
```

## AWS Bedrock Integration

### Authentication Flow

```
1. AWS_PROFILE=ai-learning in environment
         │
2. boto3.Session(profile_name='ai-learning')
         │
3. Loads from ~/.aws/credentials
         │
4. Signs requests with AWS Signature V4
         │
5. Calls bedrock-runtime.us-east-1.amazonaws.com
```

### Model Invocation

```python
# Non-streaming
response = client.invoke_model(
    modelId="us.anthropic.claude-sonnet-4-6",  # Inference profile ID
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7
    }),
    contentType="application/json"
)

# Streaming
response = client.invoke_model_with_response_stream(...)
for event in response['body']:
    chunk = json.loads(event['chunk']['bytes'])
    # Process chunk
```

### Why Inference Profiles?

AWS Bedrock requires inference profiles for newer models:

```
Raw Model ID:        anthropic.claude-sonnet-4-6        ❌ Doesn't work
Inference Profile:   us.anthropic.claude-sonnet-4-6    ✅ Works
```

Inference profiles provide:
- Cross-region routing
- Managed throughput
- Automatic failover

## Error Handling

### Layers

| Layer | Error Type | HTTP Status |
|-------|-----------|-------------|
| Pydantic | Validation error | 422 |
| Route | Business logic | 400 |
| Provider | AWS/API errors | 500 |
| Global | Unexpected | 500 |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

## Security Considerations

### Current (Development)

- AWS credentials via CLI profile (secure)
- No API authentication on gateway
- CORS allows all origins

### Production Recommendations

1. **Add API Key Authentication**
   ```python
   async def verify_api_key(api_key: str = Header(...)):
       if api_key not in valid_keys:
           raise HTTPException(401)
   ```

2. **Restrict CORS**
   ```python
   app.add_middleware(CORSMiddleware, allow_origins=["https://yourdomain.com"])
   ```

3. **Add Rate Limiting**
4. **Use IAM Roles** (not CLI profiles) in production
5. **Enable CloudWatch Logging**

## Extending the System

### Adding a New Provider

1. Create `providers/new_provider.py`
2. Implement `BaseLLMProvider` interface
3. Register in `LLMRouter.__init__`
4. Add pricing to `tokenizer.MODEL_PRICING`

### Adding a New Model

Edit `bedrock_provider.py`:
```python
MODEL_MAPPING = {
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "new-model": "us.provider.new-model-id",  # Add here
}
```

### Adding New Endpoints

1. Add route in `api/routes.py`
2. Define request/response models in `models.py`
3. Update README documentation

---

*Architecture documented for Week 1: AI Gateway project*
