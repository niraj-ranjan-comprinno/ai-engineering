"""
Configuration for AI Agents service.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Service info
    app_name: str = "AI Agents"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Week 1 Gateway connection (for LLM calls)
    gateway_url: str = "http://localhost:8000"
    
    # Agent settings
    max_iterations: int = 10  # Maximum ReAct loops before stopping
    default_model: str = "claude-sonnet-4-6"  # Use gateway's model alias
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()
