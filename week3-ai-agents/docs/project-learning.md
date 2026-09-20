# Week 3: AI Agents - Deep Dive Learning Guide

## Table of Contents
1. [What is an AI Agent?](#what-is-an-ai-agent)
2. [The ReAct Pattern](#the-react-pattern)
3. [Function Calling / Tool Use](#function-calling--tool-use)
4. [Building Tools](#building-tools)
5. [Memory Management](#memory-management)
6. [Putting It All Together](#putting-it-all-together)

---

## What is an AI Agent?

### Chatbot vs Agent

**Chatbot (Week 1-2 style):**
```
User → LLM → Response
```
One step. LLM generates text based on input. No actions.

**Agent (Week 3):**
```
User → LLM → [Think] → [Act] → [Observe] → [Think] → ... → Response
```
Multiple steps. LLM reasons, takes actions, observes results, repeats.

### The Key Difference: Agency

| Chatbot | Agent |
|---------|-------|
| Generates text | Takes actions |
| Knowledge frozen at training | Accesses real-time data |
| One-shot response | Multi-step reasoning |
| Can't verify information | Can look things up |
| Passive | Active |

### Real-World Example

**Without Agent:**
```
User: "What's 23.7% of $1,847?"
LLM: "23.7% of $1,847 is approximately $437.74"  
     ← May be wrong! LLMs are bad at math
```

**With Agent:**
```
User: "What's 23.7% of $1,847?"
Agent: [Thinking] I need to calculate this precisely
       [Action] calculator("1847 * 0.237")
       [Observation] Result: 437.739
       [Response] "23.7% of $1,847 is exactly $437.74"
       ← Guaranteed correct!
```

---

## The ReAct Pattern

### What is ReAct?

**ReAct = Reasoning + Acting**

From the 2022 paper by Yao et al., ReAct interleaves:
- **Reasoning**: LLM thinks about what to do
- **Acting**: LLM executes a tool
- **Observing**: Results fed back to LLM

### The Loop

```
┌─────────────────────────────────────────┐
│                                         │
│  ┌─────────┐                            │
│  │  THINK  │  "I need to calculate..."  │
│  └────┬────┘                            │
│       │                                 │
│       ▼                                 │
│  ┌─────────┐                            │
│  │   ACT   │  calculator("2 + 2")       │
│  └────┬────┘                            │
│       │                                 │
│       ▼                                 │
│  ┌─────────┐                            │
│  │ OBSERVE │  Result: 4                 │
│  └────┬────┘                            │
│       │                                 │
│       ▼                                 │
│  ┌─────────┐                            │
│  │  THINK  │  "Now I can answer..."     │
│  └────┬────┘                            │
│       │                                 │
│       ▼                                 │
│  ┌─────────┐                            │
│  │ RESPOND │  "The answer is 4"         │
│  └─────────┘                            │
│                                         │
└─────────────────────────────────────────┘
```

### Code Implementation

```python
class ReActAgent:
    async def run(self, user_input: str) -> AgentResponse:
        while iteration < max_iterations:
            # 1. THINK + ACT: LLM decides what to do
            llm_response = await self._call_llm_with_tools()
            
            if llm_response.get("tool_use"):
                # 2. Execute the tool
                tool_result = await self._execute_tool(
                    tool_name, tool_input
                )
                
                # 3. OBSERVE: Feed result back to LLM
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Tool returned: {tool_result}"
                })
                
            else:
                # LLM is done, return final answer
                return AgentResponse(answer=llm_response["content"])
```

### Why Max Iterations?

Agents can get stuck in loops. Safety limit prevents:
- Infinite tool calling
- Runaway costs
- Hung requests

Default: 10 iterations. Adjust based on task complexity.

---

## Function Calling / Tool Use

### How Claude Uses Tools

When you send tools to Claude, it can respond in two ways:

**1. Text Response (no tool needed):**
```json
{
  "content": [
    {"type": "text", "text": "Hello! How can I help?"}
  ],
  "stop_reason": "end_turn"
}
```

**2. Tool Use Request:**
```json
{
  "content": [
    {"type": "text", "text": "Let me calculate that..."},
    {
      "type": "tool_use",
      "id": "toolu_01ABC...",
      "name": "calculator",
      "input": {"expression": "25 * 180 / 100"}
    }
  ],
  "stop_reason": "tool_use"
}
```

### Tool Schema Format

Claude expects tools in this format:

```python
{
    "name": "calculator",
    "description": "Perform math calculations...",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}
```

### The Description Matters!

The LLM reads the description to decide when to use a tool:

**Bad description:**
```python
description = "Calculator"
# LLM doesn't know when or how to use it
```

**Good description:**
```python
description = """Perform mathematical calculations.
Use this when you need to do arithmetic, percentages, or complex math.
Input should be a valid expression like '2 + 2' or 'sqrt(16)'.
Supported: +, -, *, /, **, sqrt(), sin(), cos(), log()"""
# LLM knows exactly when and how to use it
```

---

## Building Tools

### The Tool Base Class

Every tool inherits from `Tool`:

```python
class Tool(ABC):
    name: str          # Unique identifier
    description: str   # What it does (LLM reads this!)
    parameters: dict   # JSON Schema for inputs
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """The actual implementation"""
        pass
```

### Creating a Custom Tool

```python
class StockPriceTool(Tool):
    name = "get_stock_price"
    description = """Get current stock price by ticker symbol.
Use when user asks about stock prices, market data, or company valuations.
Example tickers: AAPL, GOOGL, MSFT, AMZN"""
    
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., 'AAPL')"
            }
        },
        "required": ["ticker"]
    }
    
    async def execute(self, ticker: str) -> ToolResult:
        # Call a real stock API here
        price = await fetch_stock_price(ticker)
        return ToolResult(
            success=True,
            result={"ticker": ticker, "price": price}
        )
```

### Tool Result Format

```python
@dataclass
class ToolResult:
    success: bool     # Did it work?
    result: Any       # The data
    error: str | None # Error message if failed
    
    def to_observation(self) -> str:
        """Format for LLM to read"""
        if self.success:
            return f"Tool succeeded. Result: {self.result}"
        else:
            return f"Tool failed: {self.error}"
```

### Tool Design Principles

1. **Single Responsibility**: One tool = one job
2. **Clear Descriptions**: LLM must understand when to use it
3. **Robust Error Handling**: Return errors, don't crash
4. **Typed Parameters**: Use JSON Schema for validation
5. **Idempotent**: Same input → same output

---

## Memory Management

### Types of Memory

```
┌─────────────────────────────────────────────────────────┐
│                     MEMORY TYPES                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  SHORT-TERM (Conversation)     LONG-TERM (Persistent)   │
│  ─────────────────────────     ─────────────────────    │
│  • Current chat session        • Across sessions         │
│  • Stored in RAM               • Stored in DB/vector     │
│  • Lost when session ends      • "Remember when..."      │
│  • This week's focus           • Week 2's vector store!  │
│                                                          │
│  WORKING MEMORY                                          │
│  ──────────────                                          │
│  • Current task context                                  │
│  • Tool results being processed                          │
│  • Intermediate reasoning                                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Conversation Memory

```python
class ConversationMemory:
    max_messages: int = 100
    messages: list[Message]
    
    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))
        self._trim_if_needed()
    
    def add_assistant_message(self, content: str):
        self.messages.append(Message(role="assistant", content=content))
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Remove old messages to stay under limit"""
        if len(self.messages) > self.max_messages:
            # Keep system prompt, remove oldest others
            self.messages = system_msgs + other_msgs[-keep_count:]
```

### Why Trim Messages?

1. **Context Window Limits**: LLMs have max tokens (128K for Claude)
2. **Cost**: More tokens = more expensive
3. **Performance**: Smaller context = faster inference
4. **Relevance**: Recent messages usually matter more

### Session Management

```python
class SessionManager:
    """Manage multiple conversation sessions"""
    
    def get_or_create(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory()
        return self._sessions[session_id]
```

Use cases:
- Multi-user applications
- Separate conversations per topic
- Persistent sessions

---

## Putting It All Together

### Request Flow

```
1. User sends: "What's 25% of 180?"
   
2. Agent receives request
   └── Creates/loads session memory
   └── Initializes ReAct agent with tools

3. ReAct Loop - Iteration 1:
   └── Send to LLM with tools
   └── LLM returns: tool_use(calculator, "180 * 0.25")
   └── Execute calculator → 45.0
   └── Feed observation back to LLM

4. ReAct Loop - Iteration 2:
   └── Send observation to LLM
   └── LLM returns: text("25% of 180 is 45")
   └── No tool_use → done!

5. Return response:
   {
     "answer": "25% of 180 is 45",
     "tools_used": ["calculator"],
     "iterations": 2
   }
```

### Code Flow

```python
# routes.py
@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    # 1. Get session memory
    memory = session_manager.get_or_create(request.session_id)
    
    # 2. Create agent
    agent = ReActAgent(tool_registry=tool_registry)
    agent.conversation_history = memory.get_messages()
    
    # 3. Run agent (ReAct loop happens here)
    result = await agent.run(request.message)
    
    # 4. Update memory
    memory.add_user_message(request.message)
    memory.add_assistant_message(result.answer)
    
    # 5. Return response
    return ChatResponse(
        answer=result.answer,
        tools_used=result.tools_used,
        iterations=result.total_iterations
    )
```

---

## Key Takeaways

1. **Agents = LLM + Tools + Loop**
   - LLMs gain capabilities through tools
   - ReAct pattern enables multi-step reasoning

2. **Tool Design is Critical**
   - Clear descriptions help LLM choose correctly
   - Robust error handling prevents crashes

3. **Memory Enables Continuity**
   - Session memory for multi-turn conversations
   - Trim old messages to manage costs

4. **Function Calling is Native**
   - Claude has built-in tool use support
   - JSON schemas define tool interfaces

5. **Safety Limits Matter**
   - Max iterations prevent infinite loops
   - Cost tracking prevents runaway spending

---

## What's Next?

**Week 4: Evaluation & Testing**
- How do you know if your agent is working well?
- Automated testing for LLM outputs
- Evaluation metrics and benchmarks

**Future Enhancements:**
- Add more sophisticated tools (database queries, API calls)
- Implement long-term memory with vector store
- Build multi-agent systems
- Add planning and task decomposition
