"""
Agent Core - The ReAct (Reasoning + Acting) Agent Implementation.

KEY CONCEPT: What is an Agent?
==============================
An agent is an LLM that can:
1. REASON about what to do (think)
2. ACT by using tools (do)
3. OBSERVE the results (see)
4. REPEAT until the task is complete

This is fundamentally different from a simple chatbot:
- Chatbot: User → LLM → Response (one step)
- Agent: User → LLM → Tool → Observe → LLM → Tool → ... → Response (multi-step)

THE ReAct PATTERN
=================
ReAct = Reasoning + Acting (from the 2022 paper by Yao et al.)

Loop:
┌─────────────────────────────────────────────────────────┐
│  1. THINK: "I need to calculate 15% of 250..."          │
│  2. ACT: Call calculator tool with "250 * 0.15"         │
│  3. OBSERVE: Tool returns 37.5                          │
│  4. THINK: "The answer is 37.5. I can respond now."     │
│  5. RESPOND: "15% of 250 is 37.5"                       │
└─────────────────────────────────────────────────────────┘

The LLM decides when to use tools and when to stop.
We provide tools via function calling (Claude's native feature).
"""
import json
import httpx
from typing import AsyncGenerator
from dataclasses import dataclass, field

from ..config import get_settings
from ..tools import ToolRegistry, ToolResult


@dataclass
class AgentStep:
    """
    Represents one step in the agent's execution.
    
    Each step can be:
    - A thought (reasoning)
    - A tool call (action)
    - A tool result (observation)
    - A final response
    """
    step_type: str  # "thought", "tool_call", "tool_result", "response"
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_result: ToolResult | None = None


@dataclass
class AgentResponse:
    """Complete response from agent execution."""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    tools_used: list[str] = field(default_factory=list)


class ReActAgent:
    """
    ReAct Agent - Combines reasoning and acting with tool use.
    
    The agent:
    1. Receives a user query
    2. Sends query + available tools to LLM
    3. LLM either responds OR requests a tool call
    4. If tool call: execute tool, feed result back to LLM
    5. Repeat until LLM gives final response
    
    Usage:
        agent = ReActAgent(tool_registry)
        response = await agent.run("What's 25% of 180?")
        print(response.answer)  # "25% of 180 is 45"
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_iterations: int | None = None,
        model: str | None = None,
    ):
        """
        Initialize the agent.
        
        Args:
            tool_registry: Registry of available tools
            max_iterations: Max tool-use loops (prevents infinite loops)
            model: LLM model to use
        """
        self.settings = get_settings()
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations or self.settings.max_iterations
        self.model = model or self.settings.default_model
        self.gateway_url = self.settings.gateway_url
        
        # Conversation history for context
        self.conversation_history: list[dict] = []
    
    async def run(self, user_input: str) -> AgentResponse:
        """
        Run the agent on a user query.
        
        This is the main entry point. It implements the ReAct loop:
        Think → Act → Observe → Repeat
        
        Args:
            user_input: The user's question/request
            
        Returns:
            AgentResponse with answer and execution trace
        """
        steps: list[AgentStep] = []
        tools_used: list[str] = []
        iteration = 0
        
        # Track tool results for this request to avoid loops
        tool_results_this_request: list[dict] = []
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{self.max_iterations}")
            
            # Call LLM with tools and accumulated tool results
            llm_response = await self._call_llm_with_tools(tool_results_this_request)
            
            # Check if LLM wants to use a tool
            if llm_response.get("tool_use"):
                tool_call = llm_response["tool_use"]
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_id = tool_call.get("id", f"call_{iteration}")
                
                print(f"🔧 Tool call: {tool_name}({tool_input})")
                
                # Record the tool call step
                steps.append(AgentStep(
                    step_type="tool_call",
                    content=f"Calling {tool_name}",
                    tool_name=tool_name,
                    tool_input=tool_input
                ))
                
                # Execute the tool
                tool_result = await self._execute_tool(tool_name, tool_input)
                tools_used.append(tool_name)
                
                result_str = json.dumps(tool_result.result) if tool_result.success else tool_result.error
                print(f"📋 Tool result: {result_str[:200]}...")
                
                # Record the observation step
                steps.append(AgentStep(
                    step_type="tool_result",
                    content=tool_result.to_observation(),
                    tool_name=tool_name,
                    tool_result=tool_result
                ))
                
                # Add to tool results for next LLM call
                # This format matches Claude's expected tool_result format
                tool_results_this_request.append({
                    "tool_use_id": tool_id,
                    "tool_name": tool_name,
                    "result": result_str
                })
                
            else:
                # LLM gave a final response (no tool call)
                final_answer = llm_response.get("content", "")
                print(f"✅ Final answer: {final_answer[:100]}...")
                
                steps.append(AgentStep(
                    step_type="response",
                    content=final_answer
                ))
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_answer
                })
                
                return AgentResponse(
                    answer=final_answer,
                    steps=steps,
                    total_iterations=iteration,
                    tools_used=list(set(tools_used))
                )
        
        # Max iterations reached - provide partial answer if we have tool results
        if tool_results_this_request:
            partial_answer = "I gathered some information but couldn't complete the task:\n\n"
            for tr in tool_results_this_request:
                partial_answer += f"- {tr['tool_name']}: {tr['result']}\n"
        else:
            partial_answer = "I wasn't able to complete this task within the allowed steps. Please try rephrasing your request."
        
        return AgentResponse(
            answer=partial_answer,
            steps=steps,
            total_iterations=iteration,
            tools_used=list(set(tools_used))
        )
    
    async def run_stream(self, user_input: str) -> AsyncGenerator[AgentStep, None]:
        """
        Stream agent execution step by step.
        
        Yields each step as it happens, allowing real-time UI updates.
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        tool_results_this_request: list[dict] = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            yield AgentStep(
                step_type="thought",
                content=f"Thinking... (iteration {iteration})"
            )
            
            llm_response = await self._call_llm_with_tools(tool_results_this_request)
            
            if llm_response.get("tool_use"):
                tool_call = llm_response["tool_use"]
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_id = tool_call.get("id", f"call_{iteration}")
                
                yield AgentStep(
                    step_type="tool_call",
                    content=f"Using tool: {tool_name}",
                    tool_name=tool_name,
                    tool_input=tool_input
                )
                
                tool_result = await self._execute_tool(tool_name, tool_input)
                result_str = json.dumps(tool_result.result) if tool_result.success else tool_result.error
                
                yield AgentStep(
                    step_type="tool_result",
                    content=tool_result.to_observation(),
                    tool_name=tool_name,
                    tool_result=tool_result
                )
                
                tool_results_this_request.append({
                    "tool_use_id": tool_id,
                    "tool_name": tool_name,
                    "result": result_str
                })
                
            else:
                final_answer = llm_response.get("content", "")
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_answer
                })
                
                yield AgentStep(
                    step_type="response",
                    content=final_answer
                )
                return
        
        yield AgentStep(
            step_type="response",
            content="Max iterations reached. Please try again."
        )
    
    async def _call_llm_with_tools(self, tool_results: list[dict]) -> dict:
        """
        Call the LLM (via Week 1 Gateway) with tool schemas.
        
        The LLM will either:
        1. Return a text response (done)
        2. Return a tool_use request (need to execute tool)
        
        Args:
            tool_results: List of tool results from previous iterations
        """
        tools = self.tool_registry.get_all_schemas()
        
        # Improved system prompt with clearer instructions
        system_prompt = """You are a helpful AI assistant with access to tools.

IMPORTANT RULES:
1. When you need to perform calculations, get dates/times, check weather, or search for information - USE THE AVAILABLE TOOLS.
2. Do NOT guess or make up information - use tools to get accurate data.
3. After using a tool and receiving a result, IMMEDIATELY provide your final answer to the user. Do NOT call the same tool again.
4. If you need multiple pieces of information, call each required tool ONCE, then combine the results in your answer.
5. Be concise and helpful in your responses.

WORKFLOW:
- Analyze what the user needs
- Call necessary tools (each tool only once per piece of information)
- Once you have the tool results, formulate your final answer
- Do NOT repeat tool calls - use the results you already have"""

        # Build messages list
        messages = []
        
        # Add conversation history (user messages and previous assistant responses)
        for msg in self.conversation_history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant" and msg.get("content"):
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # If we have tool results from this request, add them as context
        # This is the key fix - we include tool results in a way the LLM can understand
        if tool_results:
            # Build a summary of tool results for context
            tool_context = "\n\n[TOOL RESULTS - Use these to answer the user's question, do NOT call these tools again]:\n"
            for tr in tool_results:
                tool_context += f"- {tr['tool_name']} returned: {tr['result']}\n"
            tool_context += "\nNow provide your final answer based on these results."
            
            # Add the tool context as a follow-up user message
            messages.append({"role": "user", "content": tool_context})
        
        request_body = {
            "model": self.model,
            "messages": messages,
            "system": system_prompt,
            "tools": tools,
            "max_tokens": 1024,
            "temperature": 0.3,  # Lower temperature for more consistent behavior
            "stream": False,
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=request_body
            )
            response.raise_for_status()
            data = response.json()
        
        # Parse the response
        content = data.get("content", [])
        
        result = {"content": "", "tool_use": None}
        
        for block in content:
            if block.get("type") == "text":
                result["content"] = block.get("text", "")
            elif block.get("type") == "tool_use":
                result["tool_use"] = {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input", {})
                }
        
        return result
    
    async def _execute_tool(self, tool_name: str, tool_input: dict) -> ToolResult:
        """
        Execute a tool by name with given input.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Parameters to pass to the tool
            
        Returns:
            ToolResult from the tool execution
        """
        tool = self.tool_registry.get(tool_name)
        
        if tool is None:
            return ToolResult(
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}"
            )
        
        try:
            return await tool.execute(**tool_input)
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool execution failed: {str(e)}"
            )
    
    def reset_conversation(self):
        """Clear conversation history for a fresh start."""
        self.conversation_history = []
