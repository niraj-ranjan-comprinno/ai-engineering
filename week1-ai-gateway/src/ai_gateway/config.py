"""
Configuration Management using Pydantic Settings.

=============================================================================
THEORY: Environment-Based Configuration
=============================================================================

Why use environment variables for configuration?
1. Security: API keys never go into source code
2. Flexibility: Different configs for dev/staging/prod without code changes
3. 12-Factor App: Industry standard for cloud-native applications

Pydantic Settings provides:
- Type validation for config values
- Automatic .env file loading
- Clear documentation of required settings
- Default values with override capability
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic will automatically:
    1. Look for a .env file in the current directory
    2. Load environment variables
    3. Validate types and provide defaults
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # ==========================================================================
    # API Keys - Set the ones you want to use
    # ==========================================================================
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # ==========================================================================
    # AWS Bedrock Configuration
    # ==========================================================================
    # Option 1: Use Bedrock API Key (simpler!)
    bedrock_api_key: str = ""
    
    # Option 2: Use AWS credentials (set use_bedrock=true, no api key)
    use_bedrock: bool = False
    aws_region: str = "us-east-1"
    aws_profile: str = ""  # Optional: AWS CLI profile name
    
    # ==========================================================================
    # Provider Configuration
    # ==========================================================================
    # Which LLM provider to use by default: openai, anthropic, or bedrock
    default_provider: Literal["openai", "anthropic", "bedrock"] = "openai"
    
    # Default model for each provider
    default_model: str = "gpt-4o-mini"
    
    # ==========================================================================
    # Application Settings
    # ==========================================================================
    log_level: str = "INFO"
    rate_limit_rpm: int = 60  # Requests per minute
    enable_cost_tracking: bool = True
    
    # ==========================================================================
    # Server Configuration
    # ==========================================================================
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    THEORY: @lru_cache Decorator
    ----------------------------
    lru_cache (Least Recently Used cache) memoizes the function result.
    This means Settings() is only instantiated ONCE, and subsequent calls
    return the cached instance. This is a simple singleton pattern.
    
    Why cache settings?
    1. .env file is read only once at startup
    2. Validation happens only once
    3. All parts of the app share the same config
    """
    return Settings()
