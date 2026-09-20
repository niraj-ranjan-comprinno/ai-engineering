# Week 1: The AI Gateway

> A streaming LLM backend with real-time token usage and cost tracking.

**Provider**: AWS Bedrock  
**Model**: Claude Sonnet 4.6  
**Authentication**: AWS CLI Profile (`ai-learning`)

## 🎯 Learning Outcomes

By completing this project, you will understand:

- **Python Fundamentals for AI**: Async/await, type hints, dataclasses, ABC patterns
- **Async Python & APIs**: Non-blocking I/O, httpx, FastAPI
- **Streaming Responses**: Server-Sent Events (SSE), chunk processing
- **Transformer Internals**: Tokenization basics, BPE encoding
- **LLM Lifecycle**: Request/response flow, token economics, cost optimization
- **AWS Bedrock**: Inference profiles, model invocation, boto3 integration

## 📁 Project Structure

```
week1-ai-gateway/
├── src/ai_gateway/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # FastAPI application entry point
│   ├── config.py             # Pydantic settings management
│   ├── models.py             # Request/response schemas
│   ├── tokenizer.py          # Token counting & cost calculation
│   ├── api/
│   │   ├── routes.py         # API endpoints
│   │   └── middleware.py     # Logging & metrics
│   └── providers/
│       ├── base.py           # Abstract provider interface
│       ├── bedrock_provider.py   # AWS Bedrock (current)
│       ├── openai_provider.py    # OpenAI (optional)
│       └── router.py         # Intelligent routing
├── docs/
│   ├── project-learning.md   # Complete learning guide
│   ├── learning1.md          # Bedrock integration lessons
│   ├── THEORY.md             # Deep dive into concepts
│   └── ARCHITECTURE.md       # System design
├── examples/
│   ├── simple_request.py     # Basic API usage
│   └── streaming_example.py  # SSE consumption
├── scripts/
│   ├── client.py             # Interactive CLI client
│   └── test_api.sh           # curl-based API tests
├── pyproject.toml            # Python packaging
├── .env                      # Configuration (secrets - not in git)
└── README.md                 # You are here!
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- AWS CLI configured with profile `ai-learning`
- AWS Bedrock access enabled

### 2. Set Up Environment

```bash
# Navigate to project
cd week1-ai-gateway

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install boto3  # For AWS Bedrock
```

### 3. Configure AWS Profile

```bash
# Set up AWS credentials (one-time)
aws configure --profile ai-learning

# Verify it works
aws sts get-caller-identity --profile ai-learning
```

### 4. Run the Server

```bash
# Start with AWS profile
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload

# Server runs at http://localhost:8000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/v1/health

# List providers
curl http://localhost:8000/v1/providers

# Chat completion (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": false}'

# Chat completion (streaming)
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Count to 5"}], "stream": true}'
```

### 6. Interactive Client

```bash
# Run the CLI client
python scripts/client.py

# Or demo mode
python scripts/client.py --demo
```

## 📚 Configuration

### Current Setup (`.env`)

```bash
# AWS Bedrock Configuration
USE_BEDROCK=true
AWS_REGION=us-east-1
AWS_PROFILE=ai-learning

# Gateway Configuration
DEFAULT_PROVIDER=bedrock
DEFAULT_MODEL=claude-sonnet-4-6
LOG_LEVEL=INFO
ENABLE_COST_TRACKING=true
```

### Available Models

| Model Name | Inference Profile ID | Description |
|------------|---------------------|-------------|
| `claude-sonnet-4-6` | `us.anthropic.claude-sonnet-4-6` | Claude Sonnet 4.6 (recommended) |

## 🔧 API Reference

### Chat Completion

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "model": "claude-sonnet-4-6",  // optional
  "temperature": 0.7,            // optional (0.0-2.0)
  "max_tokens": 1000,            // optional
  "stream": true                 // optional (default: true)
}
```

**Response (non-streaming):**
```json
{
  "id": "uuid",
  "content": "Hello! How can I help?",
  "model": "claude-sonnet-4-6",
  "provider": "bedrock",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23
  },
  "cost": {
    "input_cost": 0.000015,
    "output_cost": 0.000024,
    "total_cost": 0.000039
  },
  "latency_ms": 1523.4
}
```

**Response (streaming - SSE):**
```
event: chunk
data: {"id":"uuid","delta":"Hello","finish_reason":null}

event: chunk
data: {"id":"uuid","delta":"!","finish_reason":null}

event: complete
data: {"id":"uuid","usage":{...},"cost":{...},"latency_ms":1234}

event: done
data: [DONE]
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/providers` | GET | List available providers |
| `/v1/tokens/count` | POST | Count tokens without calling LLM |
| `/metrics` | GET | View request metrics |

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [project-learning.md](docs/project-learning.md) | Complete beginner's guide |
| [learning1.md](docs/learning1.md) | Bedrock integration lessons learned |
| [THEORY.md](docs/THEORY.md) | Deep dive into concepts |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design & data flow |

## ✅ Week 1 Checklist

- [ ] Set up the project and run the server
- [ ] Make your first API call with curl
- [ ] Understand how streaming works (watch the SSE events)
- [ ] Try the interactive client
- [ ] Read `docs/project-learning.md`
- [ ] Complete the hands-on exercises
- [ ] Experiment with temperature settings
- [ ] Track costs for a conversation

## 🔜 Next Week Preview

**Week 2: RAG Foundations - The Knowledge Engine**

You'll build a context-aware RAG pipeline that:
- Converts documents to embeddings
- Stores vectors in a local database
- Retrieves relevant context for queries
- Generates accurate, grounded responses

The AI Gateway you built this week will be the foundation for calling LLMs in the RAG pipeline!

---

*Built as part of the AI Engineering learning path.*
