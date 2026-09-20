# Week 2: RAG Foundations - Learning Guide

## For Software Engineers Learning AI Engineering

---

## 📌 What is This Project?

This project is a **RAG (Retrieval-Augmented Generation) Engine** - a system that enables AI to answer questions using YOUR documents as a knowledge source.

```
Traditional LLM:
    Question → LLM → Answer (based on training data only, may hallucinate)

RAG System:
    Question → Search Your Docs → Relevant Context → LLM → Grounded Answer
```

**Think of it as:** Giving the AI access to your company's documentation, letting it answer questions accurately with citations.

---

## 🎯 Problem Statement

### The Challenge

You want to build a Q&A system over your company's internal documents:
- Policy manuals
- Technical documentation
- Knowledge base articles
- Meeting notes

**Without RAG:**
| Problem | Impact |
|---------|--------|
| LLM doesn't know your docs | Can't answer company-specific questions |
| Hallucination | Makes up plausible-sounding but wrong answers |
| No citations | Can't verify where answer came from |
| Knowledge cutoff | Training data is outdated |
| Fine-tuning is expensive | $$$$ and takes days |

### The Solution: RAG

RAG adds a retrieval step before generation:

1. ✅ **Ingest your documents** → Store as vector embeddings
2. ✅ **User asks question** → Convert to embedding
3. ✅ **Search for similar content** → Find relevant chunks
4. ✅ **Augment the prompt** → Add retrieved context
5. ✅ **Generate answer** → LLM answers based on YOUR docs
6. ✅ **Cite sources** → User knows where info came from

---

## 🚀 How to Run This Project

### Prerequisites

```bash
# Python 3.11+
python3 --version

# AWS CLI configured
aws sts get-caller-identity --profile ai-learning

# Week 1 AI Gateway running
curl http://localhost:8000/v1/health
```

### Step 1: Navigate to Project

```bash
cd /Users/nirajranjan/Documents/developer/learning/week2-rag-foundations
```

### Step 2: Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Step 3: Start Week 1 AI Gateway (in another terminal)

```bash
cd ../week1-ai-gateway
source .venv/bin/activate
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload --port 8000
```

### Step 4: Start RAG Engine

```bash
AWS_PROFILE=ai-learning PYTHONPATH=src uvicorn rag_engine.main:app --reload --port 8001
```

You should see:
```
INFO:     RAG Engine starting up...
INFO:     ChromaDB: ./data/chroma
INFO:     AI Gateway: http://localhost:8000
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### Step 5: Ingest Sample Documents

```bash
curl -X POST "http://localhost:8001/v1/ingest/directory?dir_path=./data/documents"
```

### Step 6: Ask a Question

```bash
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?", "stream": false}'
```

---

## 📚 What Knowledge You Will Gain

| Skill Area | What You'll Learn |
|------------|-------------------|
| **Embeddings** | What they are, how they capture meaning |
| **Vector Databases** | ChromaDB, similarity search, HNSW |
| **Document Processing** | Chunking strategies, overlap, metadata |
| **RAG Architecture** | Pipeline design, prompt engineering |
| **Retrieval** | Top-K, similarity thresholds, re-ranking |
| **Integration** | Connecting services (Gateway + RAG) |

---

## 📖 Detailed Learning Topics

---

# Topic 1: What are Embeddings?

## The Core Concept

Embeddings convert text into **dense vectors** (lists of numbers) that capture semantic meaning.

```
"The cat sat on the mat"
         ↓ Embedding Model
[0.023, -0.156, 0.892, ..., 0.045]  (1024 numbers)
```

## Why Embeddings Work

Similar meanings → Similar vectors → Close in vector space

```
embed("happy") ≈ [0.8, 0.2, ...]
embed("joyful") ≈ [0.79, 0.21, ...]  ← Very similar!
embed("computer") ≈ [-0.3, 0.9, ...]  ← Very different
```

## Visual Representation

Imagine a 2D map (real embeddings are 1024D):

```
         happy ● ● joyful
                
    sad ●           
                    ● computer
                    ● laptop
```

Points close together = Similar meaning!

## How We Use Them

1. **Document Ingestion**: Embed each chunk → Store in vector DB
2. **Query Time**: Embed user's question → Find similar chunks

### 💻 Hands-On Exercise 1.1: Understand Embeddings

```bash
# Check embedding dimensions
curl -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python programming", "top_k": 3, "min_similarity": 0.2}'
```

**Questions:**
1. What similarity scores did you get?
2. Are the results semantically related to your query?

### 📁 Code to Study

**File:** `src/rag_engine/core/embeddings.py`

Key concepts:
- `BedrockEmbeddings` class
- `embed_text()` and `embed_texts()` methods
- Cosine similarity calculation

---

# Topic 2: Document Chunking

## Why Chunk Documents?

LLMs have context limits and work better with focused content:

```
❌ Full 50-page document → Too long, dilutes relevance
✅ Relevant paragraphs → Focused, relevant context
```

## Chunking Strategy

**Recursive Character Splitting:**
1. Try to split by paragraphs (`\n\n`)
2. If paragraph too big, split by sentences (`. `)
3. If sentence too big, split by words (` `)
4. Last resort: split by character

## The Overlap Problem

Without overlap, you lose context at boundaries:

```
Document: "Python was created by Guido van Rossum in 1991."

WITHOUT overlap:
  Chunk 1: "Python was created by"
  Chunk 2: "Guido van Rossum in 1991."
  
  Query: "Who created Python?" → Might miss the connection!

WITH overlap:
  Chunk 1: "Python was created by Guido van Rossum"
  Chunk 2: "created by Guido van Rossum in 1991."
  
  Query: "Who created Python?" → Both chunks have the answer!
```

## Chunking Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `chunk_size` | 1000 chars | Size of each chunk |
| `chunk_overlap` | 200 chars | How much chunks overlap |

**Trade-offs:**
- Smaller chunks → More precise retrieval, but may lose context
- Larger chunks → More context, but may dilute relevance
- More overlap → Better boundary handling, but more redundancy

### 💻 Hands-On Exercise 2.1: See Chunking in Action

```bash
# Check how many chunks were created
curl http://localhost:8001/v1/knowledge-base/stats
```

**Questions:**
1. How many chunks were created from 3 documents?
2. What's the average chunks per document?

### 📁 Code to Study

**File:** `src/rag_engine/core/chunker.py`

Look at:
- `TextChunker` class
- `_recursive_split()` method
- `_apply_overlap()` method

---

# Topic 3: Vector Databases

## What is a Vector Database?

A database optimized for storing and searching vectors (embeddings).

**Traditional DB:** Find exact matches
```sql
SELECT * FROM docs WHERE title = 'Python Tutorial'
```

**Vector DB:** Find similar items
```python
results = vector_db.search(
    query_embedding=[0.1, 0.2, ...],
    top_k=5
)
```

## How Similarity Search Works

**Brute Force (Slow):**
Compare query to EVERY vector → O(n)

**Approximate Nearest Neighbor (Fast):**
Use clever algorithms → O(log n)

## HNSW Algorithm

ChromaDB uses **HNSW** (Hierarchical Navigable Small World):

```
Layer 2:  ●─────────────●  (few nodes, long-range connections)
           \           /
Layer 1:  ●──●──●──●──●    (medium nodes)
          │  │  │  │  │
Layer 0:  ●●●●●●●●●●●●●●   (all nodes, short-range)
```

Search starts at top layer (fast jumps) and refines at lower layers.

## Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Cosine** | angle between vectors | Text embeddings (default) |
| **Euclidean** | straight-line distance | When magnitude matters |
| **Dot Product** | inner product | Normalized vectors |

We use **cosine similarity** (1 = identical, 0 = unrelated).

### 💻 Hands-On Exercise 3.1: Direct Vector Search

```bash
# Search without LLM generation
curl -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks", "top_k": 5, "min_similarity": 0.2}'
```

**Questions:**
1. What similarity scores did you get?
2. Are higher similarity results more relevant?

### 📁 Code to Study

**File:** `src/rag_engine/storage/vector_store.py`

Look at:
- `ChromaVectorStore` class
- `add_chunks()` method
- `search()` method

---

# Topic 4: The RAG Pipeline

## Complete Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. USER QUESTION                                            │
│     "What is overfitting?"                                   │
│            │                                                 │
│            ▼                                                 │
│  2. EMBED QUERY                                              │
│     [0.1, 0.2, ..., 0.9]  (1024 dims)                       │
│            │                                                 │
│            ▼                                                 │
│  3. SEARCH VECTOR DB                                         │
│     Find top-K similar chunks                                │
│            │                                                 │
│            ▼                                                 │
│  4. BUILD PROMPT                                             │
│     System: "Answer using this context: {chunks}"            │
│     User: "What is overfitting?"                             │
│            │                                                 │
│            ▼                                                 │
│  5. CALL LLM (via Week 1 Gateway)                           │
│     Generate answer grounded in context                      │
│            │                                                 │
│            ▼                                                 │
│  6. RETURN ANSWER + SOURCES                                  │
│     "Overfitting is... [Source: ml_intro.md]"               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## The RAG Prompt Template

```python
SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions 
based on the provided context.

INSTRUCTIONS:
1. Answer using ONLY the information from the context below.
2. If the context doesn't have the answer, say so.
3. Always cite your sources.

CONTEXT:
{retrieved_chunks}
"""

USER_PROMPT = """
Question: {user_question}
"""
```

## Why RAG Works

| Without RAG | With RAG |
|-------------|----------|
| LLM guesses from training | LLM answers from YOUR docs |
| May hallucinate | Grounded in real content |
| No citations | Can cite sources |
| Outdated knowledge | Always current (your docs) |

### 💻 Hands-On Exercise 4.1: Full RAG Query

```bash
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the best practices for FastAPI?",
    "top_k": 3,
    "stream": false
  }' | python3 -m json.tool
```

**Observe:**
- `answer`: Generated from context
- `sources`: Which documents were used
- `tokens_used`: Cost information

### 📁 Code to Study

**File:** `src/rag_engine/core/rag_pipeline.py`

Look at:
- `RAGPipeline` class
- `query()` method
- `RAG_SYSTEM_PROMPT` template

---

# Topic 5: Retrieval Quality

## Key Parameters

### top_k (Number of Results)

```
top_k=1:  Only most similar chunk
top_k=5:  Top 5 chunks (default)
top_k=10: More context, but may include less relevant
```

### min_similarity (Threshold)

```
min_similarity=0.9:  Very strict, few results
min_similarity=0.5:  Moderate, balanced
min_similarity=0.2:  Loose, many results
```

## Quality vs Cost Trade-off

| More Chunks | Fewer Chunks |
|-------------|--------------|
| ✅ More context | ✅ Cheaper (fewer tokens) |
| ✅ Better coverage | ✅ Faster |
| ❌ Higher cost | ❌ May miss info |
| ❌ May dilute focus | |

## Debugging Retrieval

1. **Check what's retrieved:**
   ```bash
   curl -X POST http://localhost:8001/v1/search \
     -H "Content-Type: application/json" \
     -d '{"query": "your question", "top_k": 10, "min_similarity": 0.1}'
   ```

2. **Questions to ask:**
   - Are relevant chunks being found?
   - Is similarity score reasonable?
   - Should chunk size be adjusted?

### 💻 Hands-On Exercise 5.1: Compare Retrieval Settings

```bash
# Strict (high threshold)
curl -s -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python functions", "top_k": 5, "min_similarity": 0.5}' \
  | python3 -c "import sys,json; print(f'Results: {json.load(sys.stdin)[\"total_results\"]}')"

# Loose (low threshold)
curl -s -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python functions", "top_k": 5, "min_similarity": 0.2}' \
  | python3 -c "import sys,json; print(f'Results: {json.load(sys.stdin)[\"total_results\"]}')"
```

---

# Topic 6: Document Ingestion

## Ingestion Pipeline

```
Document → Load → Chunk → Embed → Store
   │         │       │       │       │
   PDF     Parse   Split   Vector  ChromaDB
   MD      Text    1000ch  1024d   
   TXT
```

## Supported Formats

| Format | Parser | Notes |
|--------|--------|-------|
| `.txt` | Direct read | Simple |
| `.md` | Direct read | Keeps formatting |
| `.pdf` | pypdf | May lose structure |
| `.docx` | python-docx | Extracts paragraphs |
| `.html` | BeautifulSoup | Strips tags |

## Metadata

Each chunk stores:
- `source`: Original file path
- `doc_id`: Unique document identifier
- `chunk_index`: Position in document
- `doc_type`: File type
- Custom metadata

### 💻 Hands-On Exercise 6.1: Ingest Custom Text

```bash
# Ingest raw text
curl -X POST http://localhost:8001/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Kubernetes is a container orchestration platform. It manages containerized applications across multiple hosts. Key concepts include Pods, Services, and Deployments.",
    "source": "kubernetes_notes",
    "metadata": {"author": "me", "topic": "devops"}
  }'

# Now search for it
curl -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "container orchestration", "top_k": 3}'
```

### 📁 Code to Study

**File:** `src/rag_engine/core/chunker.py`

Look at:
- `DocumentLoader` class
- `load_file()` method for different formats
- `_load_pdf()`, `_load_docx()` etc.

---

# Topic 7: Integration with Week 1 Gateway

## Why Use the Gateway?

```
Option A: Direct LLM Call
  RAG → AWS Bedrock Claude → Response
  
Option B: Through Gateway (What We Do)
  RAG → AI Gateway → AWS Bedrock Claude → Response
```

**Benefits of Gateway:**
- ✅ Centralized API key management
- ✅ Cost tracking per request
- ✅ Provider abstraction (can switch models)
- ✅ Rate limiting
- ✅ Logging and metrics

## The Connection

```python
# RAG Pipeline calls Gateway
async def _call_gateway(self, system_prompt, user_prompt):
    url = f"{self._gateway_url}/v1/chat/completions"
    
    response = await self._http_client.post(url, json={
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "model": "claude-sonnet-4-6",
        "stream": False,
    })
    
    return response.json()
```

### 💻 Hands-On Exercise 7.1: Check Both Services

```bash
# Week 1 Gateway health
curl http://localhost:8000/v1/health

# Week 2 RAG health  
curl http://localhost:8001/v1/health

# Week 1 metrics (see RAG calls)
curl http://localhost:8000/metrics
```

---

## 🎯 Project Structure Reference

```
week2-rag-foundations/
├── src/rag_engine/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Settings (Pydantic)
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
│   └── chroma/              # Vector database
└── docs/
    └── project-learning.md  # This file
```

---

## ✅ Learning Checklist

- [ ] Understand what embeddings are
- [ ] Know why chunking with overlap is important
- [ ] Tested vector search without LLM
- [ ] Made full RAG queries
- [ ] Experimented with top_k and min_similarity
- [ ] Ingested custom text
- [ ] Understand how RAG connects to Week 1 Gateway
- [ ] Read the core code files

---

## 🔗 Quick Reference

### Start Both Services

```bash
# Terminal 1: Week 1 Gateway
cd week1-ai-gateway && source .venv/bin/activate
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload --port 8000

# Terminal 2: Week 2 RAG
cd week2-rag-foundations && source .venv/bin/activate
AWS_PROFILE=ai-learning PYTHONPATH=src uvicorn rag_engine.main:app --reload --port 8001
```

### Common Commands

```bash
# Ingest documents
curl -X POST "http://localhost:8001/v1/ingest/directory?dir_path=./data/documents"

# Ask question
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'

# Check stats
curl http://localhost:8001/v1/knowledge-base/stats

# Direct search
curl -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "search terms", "top_k": 5}'
```

---

## 📚 Key Concepts Summary

| Concept | What It Is |
|---------|------------|
| **Embedding** | Vector representation of text meaning |
| **Chunking** | Splitting documents into smaller pieces |
| **Vector DB** | Database for similarity search |
| **Cosine Similarity** | Measure of vector similarity (0-1) |
| **top_k** | Number of results to retrieve |
| **RAG** | Retrieval + LLM generation |

---

## 📚 Next Steps

After completing this project:

1. **Week 3: AI Agents** - Build agents with tool use
2. **Week 4: Memory Systems** - Add conversation memory
3. **Advanced RAG**: Re-ranking, hybrid search, query expansion

---

*Happy Learning! 🚀*
