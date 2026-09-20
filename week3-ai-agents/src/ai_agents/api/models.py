"""
Pydantic Models for API Request/Response schemas.
"""
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request for agent chat."""
    message: str = Field(
        description="User message to send to the agent",
        min_length=1
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity. Auto-generated if not provided."
    )


class ToolTestRequest(BaseModel):
    """Request to test a specific tool."""
    tool_name: str = Field(description="Name of the tool to test")
    parameters: dict = Field(
        default_factory=dict,
        description="Parameters to pass to the tool"
    )


# =============================================================================
# Response Models
# =============================================================================

class ToolInfo(BaseModel):
    """Information about an available tool."""
    name: str
    description: str
    parameters: dict


class StepInfo(BaseModel):
    """Information about one step in agent execution."""
    step_type: str = Field(description="Type: thought, tool_call, tool_result, response")
    content: str = Field(description="Step content or description")
    tool_name: str | None = Field(default=None, description="Tool name if applicable")
    tool_input: dict | None = Field(default=None, description="Tool input if applicable")
    tool_success: bool | None = Field(default=None, description="Tool success status")


class ChatResponse(BaseModel):
    """Response from agent chat."""
    answer: str = Field(description="Agent's final answer")
    session_id: str = Field(description="Session ID for this conversation")
    steps: list[StepInfo] = Field(
        default_factory=list,
        description="Execution trace showing agent's reasoning"
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="List of tools that were called"
    )
    iterations: int = Field(description="Number of ReAct iterations")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolTestResponse(BaseModel):
    """Response from tool test."""
    tool_name: str
    success: bool
    result: Any
    error: str | None = None


class SessionInfo(BaseModel):
    """Information about a conversation session."""
    session_id: str
    message_count: int
    created_at: datetime | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    gateway_status: str
    tools_available: int
