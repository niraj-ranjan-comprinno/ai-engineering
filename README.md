# AI Engineering Learning Path

A 10-week hands-on journey to master AI engineering concepts, built progressively with practical projects.

## 🎯 Learning Objectives

- Build production-ready AI systems from scratch
- Understand core AI/ML engineering patterns
- Gain hands-on experience with AWS Bedrock, vector databases, agents, and more

## 📚 Weekly Projects

| Week | Topic | Status | Description |
|------|-------|--------|-------------|
| 1 | [AI Gateway](./week1-ai-gateway/) | ✅ Complete | Streaming LLM backend with AWS Bedrock |
| 2 | [RAG Foundations](./week2-rag-foundations/) | ✅ Complete | Document ingestion, embeddings, vector search |
| 3 | [AI Agents](./week3-ai-agents/) | ✅ Complete | Tool use, function calling, ReAct pattern |
| 4 | Evaluation & Testing | 🔜 Next | LLM evaluation frameworks |
| 5 | Fine-tuning | 📋 Planned | Model customization techniques |
| 6 | Prompt Engineering | 📋 Planned | Advanced prompting strategies |
| 7 | Multi-modal AI | 📋 Planned | Vision and audio processing |
| 8 | Production Deployment | 📋 Planned | Scaling and monitoring |
| 9 | Security & Safety | 📋 Planned | Guardrails and content filtering |
| 10 | Capstone Project | 📋 Planned | Full-stack AI application |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Applications                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Week 3: AI Agents (Port 8002)              │
│  • ReAct Pattern         • Tool Registry                │
│  • Function Calling      • Conversation Memory          │
│  Tools: Calculator, Weather, DateTime, WebSearch        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Week 2: RAG Engine (Port 8001)             │
│  • Document Chunking     • Vector Search                │
│  • Embeddings            • Knowledge Retrieval          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Week 1: AI Gateway (Port 8000)             │
│  • Provider Abstraction  • Cost Tracking                │
│  • Streaming Support     • Function Calling             │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     AWS Bedrock                         │
│  • Claude Sonnet         • Titan Embeddings             │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- AWS CLI configured with Bedrock access (`aws sso login --profile ai-learning`)

### Running the Projects

**Week 1 - AI Gateway (required for all other weeks):**
```bash
cd week1-ai-gateway
source .venv/bin/activate
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload --port 8000
```

**Week 2 - RAG Engine:**
```bash
cd week2-rag-foundations
source .venv/bin/activate
AWS_PROFILE=ai-learning PYTHONPATH=src uvicorn rag_engine.main:app --reload --port 8001
```

**Week 3 - AI Agents:**
```bash
cd week3-ai-agents
source .venv/bin/activate
PYTHONPATH=src uvicorn ai_agents.main:app --reload --port 8002
```

### Quick Test

```bash
# Test Week 1 Gateway
curl http://localhost:8000/v1/health

# Test Week 2 RAG
curl http://localhost:8001/health

# Test Week 3 Agent
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 25% of 180?"}'
```

## 📖 Learning Resources

Each week includes:
- `README.md` - Project overview and setup
- `docs/project-learning.md` - Detailed concept explanations
- Working code with extensive comments

## 🔗 Dependencies

```
Week 3 (Agents) ──► Week 1 (Gateway) ──► AWS Bedrock
Week 2 (RAG) ────► Week 1 (Gateway) ──► AWS Bedrock
```

## 👤 Author

Learning journey by a senior software engineer transitioning to AI engineering.

## 📝 License

MIT - Feel free to use for your own learning!
