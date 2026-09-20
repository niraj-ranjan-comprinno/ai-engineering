# Week 3: AI Agents with Tool Use

Build AI agents that can **take actions** using tools to accomplish tasks.

## 🎯 Learning Objectives

- Understand the difference between chatbots and agents
- Implement the ReAct (Reasoning + Acting) pattern
- Build tools that extend LLM capabilities
- Use Claude's function calling feature
- Manage conversation memory across sessions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Request                          │
│            "What's 25% of 180?"                         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Week 3: AI Agents (Port 8002)              │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  ReAct Agent │───▶│ Tool Registry│                   │
│  │  (Think→Act) │    │  - Calculator│                   │
│  └──────────────┘    │  - Weather   │                   │
│         │            │  - DateTime  │                   │
│         │            │  - WebSearch │                   │
│         │            └──────────────┘                   │
│         │                                               │
│  ┌──────────────┐                                       │
│  │   Memory     │ ← Conversation history                │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Week 1: AI Gateway (Port 8000)             │
│              (with function calling support)            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     AWS Bedrock                         │
│                   (Claude Sonnet)                       │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Week 1 AI Gateway running on port 8000
- AWS credentials configured
- Python 3.11+

### Setup

```bash
cd week3-ai-agents

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Run

```bash
# Terminal 1: Start Week 1 Gateway (if not running)
cd ../week1-ai-gateway
source .venv/bin/activate
AWS_PROFILE=ai-learning uvicorn ai_gateway.main:app --reload --port 8000

# Terminal 2: Start Week 3 Agents
cd ../week3-ai-agents
source .venv/bin/activate
PYTHONPATH=src uvicorn ai_agents.main:app --reload --port 8002
```

### Test

```bash
# Health check
curl http://localhost:8002/health

# List available tools
curl http://localhost:8002/tools

# Test calculator tool directly
curl -X POST http://localhost:8002/tools/test \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "calculator", "parameters": {"expression": "2 + 2"}}'

# Chat with the agent
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 25% of 180?"}'
```

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with the agent (uses tools automatically) |
| `/tools` | GET | List all available tools |
| `/tools/test` | POST | Test a specific tool directly |
| `/sessions` | GET | List active conversation sessions |
| `/sessions/{id}` | DELETE | Delete a session |
| `/health` | GET | Health check |

## 🔧 Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| `calculator` | Math calculations | `"2 + 2"`, `"sqrt(16)"` |
| `get_weather` | Weather for a city | `"Tokyo"` |
| `get_datetime` | Current date/time | `"America/New_York"` |
| `web_search` | Search the web | `"Python latest version"` |

## 📁 Project Structure

```
week3-ai-agents/
├── src/ai_agents/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Request/response schemas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py         # ReAct agent implementation
│   │   └── memory.py        # Conversation memory
│   └── tools/
│       ├── __init__.py
│       ├── base.py          # Tool base class
│       ├── registry.py      # Tool registry
│       ├── calculator.py    # Calculator tool
│       ├── weather.py       # Weather tool (mock)
│       ├── datetime_tool.py # DateTime tool
│       └── web_search.py    # Web search tool (mock)
├── docs/
│   └── project-learning.md  # Detailed concepts
├── pyproject.toml
└── README.md
```

## 🧪 Example Conversations

**Simple calculation:**
```
User: "What's 15% tip on a $85 bill?"
Agent: [Uses calculator: 85 * 0.15]
       "A 15% tip on $85 would be $12.75"
```

**Weather query:**
```
User: "What's the weather in Tokyo?"
Agent: [Uses get_weather: Tokyo]
       "Tokyo is currently 22°C (72°F) and partly cloudy with 65% humidity"
```

**Multi-tool query:**
```
User: "What day is it and what's 365 * 24?"
Agent: [Uses get_datetime]
       [Uses calculator: 365 * 24]
       "Today is Sunday, September 20, 2026. 365 × 24 = 8,760 hours in a year"
```

## 📖 Key Concepts

See [docs/project-learning.md](docs/project-learning.md) for detailed explanations of:
- What makes an agent different from a chatbot
- The ReAct pattern
- Function calling / tool use
- Tool design principles
- Memory management

## 🔗 Dependencies

- **Week 1 Gateway**: Required for LLM calls
- **AWS Bedrock**: Claude Sonnet with function calling
