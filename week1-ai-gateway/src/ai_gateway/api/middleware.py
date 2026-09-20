"""
FastAPI Middleware for Request Logging and Metrics.

=============================================================================
THEORY: ASGI Middleware
=============================================================================

What is Middleware?
-------------------
Middleware wraps your application to intercept requests and responses.
It's like a pipeline:

    Request → Middleware A → Middleware B → App → Middleware B → Middleware A → Response

Each middleware can:
1. Modify the incoming request
2. Short-circuit (return early without calling app)
3. Modify the outgoing response
4. Add timing/logging around the request

ASGI Middleware:
----------------
FastAPI is built on Starlette, which uses ASGI (Async Server Gateway Interface).
ASGI middleware must:
- Be async (use async/await)
- Accept (scope, receive, send) or wrap the app callable

Two styles:
1. Pure ASGI: Low-level, full control
2. BaseHTTPMiddleware: Higher-level, easier to use (we use this)

Common Middleware Use Cases:
- Request logging and tracing
- Authentication
- CORS handling
- Rate limiting
- Response compression
- Error handling
"""

import logging
import time
import uuid
from typing import Callable
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# =============================================================================
# In-Memory Metrics Store (Use Redis/Prometheus in production!)
# =============================================================================

@dataclass
class RequestMetrics:
    """
    Metrics for a single request.
    
    THEORY: Observability Dimensions
    --------------------------------
    Good metrics have dimensions (tags) for:
    - What: endpoint, method, model, provider
    - Who: user_id, api_key, tenant
    - Where: region, server_id
    - When: timestamp
    - Result: status_code, error_type
    
    This enables queries like:
    - "Error rate for gpt-4 in the last hour"
    - "P99 latency for /chat endpoint by provider"
    - "Total tokens by user this month"
    """
    request_id: str
    method: str
    path: str
    status_code: int
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""


class MetricsStore:
    """
    Simple in-memory metrics aggregator.
    
    THEORY: Metrics Patterns
    ------------------------
    In production, use proper tools:
    - Prometheus: Pull-based metrics with powerful querying
    - StatsD/Datadog: Push-based metrics
    - OpenTelemetry: Vendor-neutral observability
    
    Key metrics for LLM applications:
    1. Request count (by status, endpoint, model)
    2. Latency distribution (P50, P90, P99)
    3. Token usage (input, output, total)
    4. Cost (by user, model, time period)
    5. Error rate (by type, endpoint)
    
    This in-memory store is for learning/development only!
    """
    
    def __init__(self):
        self.requests: list[RequestMetrics] = []
        self.totals = defaultdict(lambda: {
            "count": 0,
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
            "errors": 0,
        })
    
    def record(self, metrics: RequestMetrics):
        """Record request metrics."""
        self.requests.append(metrics)
        
        # Aggregate by endpoint
        key = f"{metrics.method}:{metrics.path}"
        self.totals[key]["count"] += 1
        self.totals[key]["total_latency_ms"] += metrics.latency_ms
        self.totals[key]["total_prompt_tokens"] += metrics.prompt_tokens
        self.totals[key]["total_completion_tokens"] += metrics.completion_tokens
        self.totals[key]["total_cost_usd"] += metrics.cost_usd
        if metrics.error:
            self.totals[key]["errors"] += 1
        
        # Keep only last 1000 requests (memory safety)
        if len(self.requests) > 1000:
            self.requests = self.requests[-1000:]
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            "endpoints": dict(self.totals),
            "recent_requests": len(self.requests),
        }


# Global metrics store (use dependency injection in production)
metrics_store = MetricsStore()


# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all requests with timing information.
    
    THEORY: Structured Logging
    --------------------------
    Structured logging outputs machine-parseable data (usually JSON):
    
    Unstructured: "Request to /chat took 150ms"
    Structured:   {"path": "/chat", "latency_ms": 150, "method": "POST"}
    
    Benefits:
    - Searchable in log aggregators (ELK, Splunk, CloudWatch)
    - Can create dashboards and alerts
    - Consistent format across services
    
    In production, use:
    - python-json-logger for JSON output
    - structlog for structured logging
    - OpenTelemetry for distributed tracing
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request with logging and timing.
        
        THEORY: Request ID (Correlation ID)
        ------------------------------------
        A unique ID for each request enables:
        - Tracing requests across services
        - Correlating logs with errors
        - Debugging specific user issues
        
        Best practice: Accept X-Request-ID from client (for distributed tracing)
        or generate one if not provided.
        """
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Add to request state for access in route handlers
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log completion
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                }
            )
            
            # Record metrics
            metrics_store.record(RequestMetrics(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2),
            ))
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "latency_ms": round(latency_ms, 2),
                }
            )
            
            # Record error metrics
            metrics_store.record(RequestMetrics(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=500,
                latency_ms=round(latency_ms, 2),
                error=str(e),
            ))
            
            raise


# =============================================================================
# Cost Tracking Middleware
# =============================================================================

class CostTrackingMiddleware(BaseHTTPMiddleware):
    """
    Track token usage and costs across requests.
    
    THEORY: Unit Economics for LLM Applications
    -------------------------------------------
    Understanding costs is critical for LLM businesses:
    
    Revenue per request - Cost per request = Margin
    
    Cost components:
    1. LLM API costs (tokens)
    2. Infrastructure (compute, storage)
    3. Observability (logging, monitoring)
    
    Key metrics:
    - Cost per conversation
    - Cost per user per month
    - Cost per feature/use case
    
    This data drives decisions like:
    - Which model to use for which tasks
    - Pricing for your product
    - Where to optimize prompts
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Track cost-related data from responses."""
        response = await call_next(request)
        
        # Cost tracking is done in the route handlers
        # This middleware could aggregate across responses
        # or integrate with a billing system
        
        return response


# =============================================================================
# Metrics Endpoint (Add to routes)
# =============================================================================

def get_metrics() -> dict:
    """
    Get current metrics.
    
    THEORY: Metrics Endpoints
    -------------------------
    Exposing metrics via HTTP enables:
    - Health check dashboards
    - Prometheus scraping
    - Admin debugging
    
    Security: In production, protect this endpoint!
    - Authentication required
    - Internal network only
    - Rate limited
    """
    return metrics_store.get_summary()
