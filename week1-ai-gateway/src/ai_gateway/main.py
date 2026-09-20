"""
FastAPI Application Entry Point.

=============================================================================
THEORY: FastAPI Application Lifecycle
=============================================================================

FastAPI Application Lifecycle:
-----------------------------
1. Import time: Module loads, decorators register routes
2. Startup: lifespan context manager runs setup
3. Running: Handles requests
4. Shutdown: lifespan cleanup runs

Lifespan Events (Modern Approach):
----------------------------------
FastAPI 0.93+ uses lifespan context managers instead of @app.on_event():

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: runs before accepting requests
        await setup_database()
        yield
        # Shutdown: runs after all requests complete
        await close_database()

Benefits:
- Resources available during entire lifespan
- Clean cleanup even on crashes
- Testable in isolation

ASGI Server (Uvicorn):
---------------------
Uvicorn is a lightning-fast ASGI server:
- Built on uvloop (fast event loop)
- HTTP/1.1 and WebSocket support
- Production-ready with workers

Running:
    uvicorn ai_gateway.main:app --reload  # Development
    uvicorn ai_gateway.main:app --workers 4  # Production
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import router, RequestLoggingMiddleware
from .api.middleware import get_metrics
from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    THEORY: Resource Management in Async Apps
    -----------------------------------------
    Async applications need careful resource management:
    
    Good patterns:
    - Create connection pools at startup
    - Share across requests (via app.state)
    - Close gracefully at shutdown
    
    Bad patterns:
    - Creating connections per-request (slow, leaks)
    - Global mutable state (race conditions)
    - Not cleaning up (resource exhaustion)
    
    The lifespan context manager ensures cleanup even if the server
    crashes or receives SIGTERM.
    """
    # Startup
    settings = get_settings()
    logger.info(f"AI Gateway starting up...")
    logger.info(f"Default provider: {settings.default_provider}")
    logger.info(f"Default model: {settings.default_model}")
    logger.info(f"Cost tracking: {'enabled' if settings.enable_cost_tracking else 'disabled'}")
    
    # You could initialize shared resources here:
    # - Database connection pools
    # - Redis clients
    # - Pre-load tokenizer models
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("AI Gateway shutting down...")
    # Clean up resources here


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="AI Gateway",
    description="""
    A streaming LLM gateway with token tracking and cost monitoring.
    
    ## Features
    - 🚀 Streaming responses via Server-Sent Events
    - 💰 Real-time token counting and cost tracking
    - 🔄 Multi-provider support (OpenAI, Anthropic)
    - 📊 Request metrics and logging
    
    ## Quick Start
    ```bash
    curl -X POST http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
    ```
    """,
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Middleware Configuration
# =============================================================================

# CORS middleware - allows cross-origin requests
# IMPORTANT: Configure properly for production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RequestLoggingMiddleware)


# =============================================================================
# Routes
# =============================================================================

# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    THEORY: API Discoverability
    ---------------------------
    Good APIs are self-documenting:
    - Root returns links to docs and health
    - OpenAPI spec auto-generated at /openapi.json
    - Interactive docs at /docs (Swagger) and /redoc
    
    This follows HATEOAS (Hypermedia as the Engine of Application State)
    principles, though most modern APIs use simpler approaches.
    """
    return {
        "service": "AI Gateway",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/v1/health",
        "providers": "/v1/providers",
    }


# Metrics endpoint (protect in production!)
@app.get("/metrics")
async def metrics():
    """
    Get application metrics.
    
    THEORY: Observability Endpoints
    -------------------------------
    Standard endpoints for observability:
    - /health: Kubernetes liveness probe
    - /ready: Kubernetes readiness probe  
    - /metrics: Prometheus scrape target
    
    In production:
    - Protect with authentication
    - Use Prometheus format
    - Add custom business metrics
    """
    return get_metrics()


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler.
    
    THEORY: Error Handling Strategy
    -------------------------------
    Errors should be:
    1. Logged with full context (for debugging)
    2. Returned with safe message (no internals exposed)
    3. Categorized by type (for monitoring)
    
    Never expose:
    - Stack traces to clients
    - Internal paths/configurations
    - Database errors verbatim
    """
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "ai_gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Auto-reload on code changes (dev only!)
    )
