"""
AI Agents Service - Week 3

This service provides AI agents with tool-use capabilities.
It connects to the Week 1 AI Gateway for LLM calls.

Architecture:
    User → AI Agents (8002) → AI Gateway (8000) → AWS Bedrock

Run with:
    uvicorn ai_agents.main:app --reload --port 8002
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup: Log configuration, verify gateway connection
    Shutdown: Clean up resources
    """
    settings = get_settings()
    
    # Startup
    logger.info("=" * 50)
    logger.info("AI Agents Service starting up...")
    logger.info(f"Gateway URL: {settings.gateway_url}")
    logger.info(f"Max iterations: {settings.max_iterations}")
    logger.info(f"Default model: {settings.default_model}")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("AI Agents Service shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Agents Service",
    description="""
Week 3: AI Agents with Tool Use

This service provides AI agents that can:
- Use tools to gather information
- Perform calculations
- Answer questions with real data
- Maintain conversation context

The agent uses the ReAct (Reasoning + Acting) pattern to:
1. Think about what to do
2. Take action (use a tool)
3. Observe the result
4. Repeat until done
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AI Agents",
        "version": "0.1.0",
        "description": "Week 3: AI Agents with Tool Use",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat - Chat with the agent",
            "tools": "GET /tools - List available tools",
            "test_tool": "POST /tools/test - Test a tool directly",
            "sessions": "GET /sessions - List active sessions",
            "health": "GET /health - Health check",
        }
    }
