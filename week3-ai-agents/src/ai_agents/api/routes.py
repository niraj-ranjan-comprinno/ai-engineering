"""
FastAPI Routes for the AI Agents service.

Endpoints:
- POST /chat - Chat with the agent (uses tools automatically)
- GET /tools - List available tools
- POST /tools/test - Test a specific tool
- GET /sessions - List active sessions
- DELETE /sessions/{session_id} - Delete a session
- GET /health - Health check
"""
import uuid
import httpx
from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..tools import ToolRegistry, register_default_tools
from ..core import ReActAgent, SessionManager

from .models import (
    ChatRequest,
    ChatResponse,
    ToolTestRequest,
    ToolTestResponse,
    ToolInfo,
    StepInfo,
    SessionInfo,
    HealthResponse,
)


# =============================================================================
# Setup
# =============================================================================

router = APIRouter()

# Initialize tool registry with default tools
tool_registry = ToolRegistry()
register_default_tools(tool_registry)

# Session manager for multi-conversation support
session_manager = SessionManager(default_max_messages=50)


# =============================================================================
# Chat Endpoint
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the AI agent.
    
    The agent will:
    1. Analyze your message
    2. Decide if tools are needed
    3. Execute tools as necessary
    4. Provide a final answer
    
    Example:
        POST /chat
        {"message": "What's 25% of 180?"}
        
    The agent will use the calculator tool and respond with "45".
    """
    settings = get_settings()
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    memory = session_manager.get_or_create(session_id)
    
    # Create agent for this request
    agent = ReActAgent(
        tool_registry=tool_registry,
        max_iterations=settings.max_iterations,
        model=settings.default_model,
    )
    
    # Copy existing conversation history to agent
    agent.conversation_history = memory.get_messages()
    
    try:
        # Run the agent
        result = await agent.run(request.message)
        
        # Update session memory
        memory.add_user_message(request.message)
        memory.add_assistant_message(result.answer)
        
        # Convert steps to response format
        steps = []
        for step in result.steps:
            steps.append(StepInfo(
                step_type=step.step_type,
                content=step.content,
                tool_name=step.tool_name,
                tool_input=step.tool_input,
                tool_success=step.tool_result.success if step.tool_result else None
            ))
        
        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            steps=steps,
            tools_used=result.tools_used,
            iterations=result.total_iterations,
        )
        
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to AI Gateway (Week 1). Make sure it's running on port 8000."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Tools Endpoints
# =============================================================================

@router.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """
    List all available tools.
    
    Returns information about each tool:
    - name: How to reference the tool
    - description: What the tool does
    - parameters: JSON schema of expected input
    """
    tools = []
    for tool in tool_registry.get_all():
        tools.append(ToolInfo(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters
        ))
    return tools


@router.post("/tools/test", response_model=ToolTestResponse)
async def test_tool(request: ToolTestRequest):
    """
    Test a specific tool directly.
    
    Useful for:
    - Verifying tool behavior
    - Debugging tool issues
    - Understanding tool output format
    
    Example:
        POST /tools/test
        {"tool_name": "calculator", "parameters": {"expression": "2 + 2"}}
    """
    tool = tool_registry.get(request.tool_name)
    
    if tool is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{request.tool_name}' not found. Available tools: {tool_registry.list_tools()}"
        )
    
    try:
        result = await tool.execute(**request.parameters)
        return ToolTestResponse(
            tool_name=request.tool_name,
            success=result.success,
            result=result.result,
            error=result.error
        )
    except Exception as e:
        return ToolTestResponse(
            tool_name=request.tool_name,
            success=False,
            result=None,
            error=str(e)
        )


# =============================================================================
# Session Management
# =============================================================================

@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """
    List all active conversation sessions.
    """
    sessions = []
    for session_id in session_manager.list_sessions():
        memory = session_manager.get(session_id)
        if memory:
            sessions.append(SessionInfo(
                session_id=session_id,
                message_count=len(memory)
            ))
    return sessions


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a conversation session.
    """
    if session_manager.delete(session_id):
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """
    Clear a session's conversation history (keep the session).
    """
    memory = session_manager.get(session_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    memory.clear()
    return {"status": "cleared", "session_id": session_id}


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Checks:
    - This service status
    - Connection to AI Gateway (Week 1)
    - Tools availability
    """
    settings = get_settings()
    
    # Check gateway connectivity
    gateway_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.gateway_url}/v1/health")
            if response.status_code == 200:
                gateway_status = "healthy"
            else:
                gateway_status = f"unhealthy (status {response.status_code})"
    except httpx.ConnectError:
        gateway_status = "unreachable"
    except Exception as e:
        gateway_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        service="ai-agents",
        gateway_status=gateway_status,
        tools_available=len(tool_registry)
    )
