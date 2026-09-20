# Week 2: RAG Foundations - The Knowledge Engine

A Retrieval-Augmented Generation (RAG) system that enables question-answering over your own documents.

## What This Project Does

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR QUESTION                                │
│                    "How do I handle errors?"                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       RAG ENGINE (Port 8001)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Embed     │───▶│   Search    │───▶│   Generate Answer       │ │
│  │   Query     │    │   VectorDB  │    │   (via AI Gateway)      │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          ANSWER                                      │
│  "Based on the FastAPI guide, you use HTTPException..."             │
│                     + SOURCE CITATIONS                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- AWS CLI configured with profile `ai-learning`
- Week 1 AI Gateway running on port 8000

### 1. Start Week 1 AI Gateway (if not running)

```bash
cd ../week1-ai-gateway
source .venv/bin/activate
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload --port 8000
```

### 2. Start RAG Engine

```bash
cd week2-rag-foundations
source .venv/bin/activate
AWS_PROFILE=ai-learning PYTHONPATH=src uvicorn rag_engine.main:app --reload --port 8001
```

### 3. Ingest Documents

```bash
# Ingest a single file
curl -X POST "http://localhost:8001/v1/ingest/file?file_path=/path/to/document.md"

# Ingest all files from a directory
curl -X POST "http://localhost:8001/v1/ingest/directory?dir_path=./data/documents"
```

### 4. Ask Questions

```bash
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is overfitting?",
    "top_k": 3,
    "stream": false
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/query` | POST | Ask a question (RAG) |
| `/v1/search` | POST | Vector search only (no LLM) |
| `/v1/ingest/text` | POST | Ingest raw text |
| `/v1/ingest/file` | POST | Ingest a file |
| `/v1/ingest/directory` | POST | Ingest all files in directory |
| `/v1/ingest/upload` | POST | Upload and ingest file |
| `/v1/knowledge-base/stats` | GET | View indexed documents |
| `/v1/knowledge-base` | DELETE | Clear all data |
| `/v1/health` | GET | Health check |

## Supported File Formats

- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.html` - HTML files

## Architecture

```
week2-rag-foundations/
├── src/rag_engine/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings management
│   ├── api/
│   │   ├── routes.py        # HTTP endpoints
│   │   └── models.py        # Request/response schemas
│   ├── core/
│   │   ├── chunker.py       # Document loading & chunking
│   │   ├── embeddings.py    # Bedrock Titan embeddings
│   │   ├── retriever.py     # Vector search
│   │   └── rag_pipeline.py  # Complete RAG pipeline
│   └── storage/
│       └── vector_store.py  # ChromaDB integration
├── data/
│   ├── documents/           # Sample documents
│   └── chroma/              # Vector database (auto-created)
└── docs/                    # Documentation
```

## Configuration

Key settings in `.env`:

```bash
# AWS Bedrock (embeddings)
AWS_PROFILE=ai-learning
AWS_REGION=us-east-1
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Week 1 AI Gateway
AI_GATEWAY_URL=http://localhost:8000

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
TOP_K_RESULTS=5
MIN_SIMILARITY=0.25
```

## Example Usage

### Query with Sources

```bash
curl -s -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Python data types?"}' | python3 -m json.tool
```

Response:
```json
{
    "answer": "Based on the Python basics guide, Python has several data types...",
    "sources": [
        {"source": "python_basics.md", "similarity": 0.72}
    ],
    "tokens_used": {"total_tokens": 630},
    "cost": {"total_cost": 0.001102}
}
```

### Direct Vector Search

```bash
curl -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "top_k": 5}'
```

### Check Knowledge Base

```bash
curl http://localhost:8001/v1/knowledge-base/stats
```

## Dependencies

- **FastAPI**: Web framework
- **ChromaDB**: Vector database
- **boto3**: AWS Bedrock SDK
- **Pydantic**: Data validation
- **httpx**: Async HTTP client

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | AWS Bedrock Titan V2 |
| Vector Store | ChromaDB |
| LLM | Claude Sonnet 4.6 (via Week 1 Gateway) |
| API Framework | FastAPI |
| Language | Python 3.11+ |

## Next Steps

- **Week 3**: Build AI Agents with tool use
- **Week 4**: Add conversation memory

---

*Part of the 10-Week AI Engineering Learning Path*
