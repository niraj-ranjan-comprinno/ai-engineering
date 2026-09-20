"""
API Layer - FastAPI routes and middleware.

=============================================================================
THEORY: API Layer Architecture
=============================================================================

The API layer handles:
1. HTTP request/response lifecycle
2. Input validation (via Pydantic)
3. Authentication/Authorization
4. Rate limiting
5. Error formatting

Separation of concerns:
-----------------------
    ┌─────────────────────────────────────┐
    │           Client Request            │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │    API Layer (routes, middleware)    │
    │    - Validation                      │
    │    - Auth                            │
    │    - Rate limiting                   │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │    Service Layer (providers, router) │
    │    - Business logic                  │
    │    - Provider abstraction            │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │    External Services (OpenAI, etc.)  │
    └─────────────────────────────────────┘

This separation makes the code:
- Testable (mock the layers below)
- Maintainable (changes isolated to one layer)
- Scalable (can add caching, queuing between layers)
"""

from .routes import router
from .middleware import RequestLoggingMiddleware, CostTrackingMiddleware

__all__ = [
    "router",
    "RequestLoggingMiddleware",
    "CostTrackingMiddleware",
]
