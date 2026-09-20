"""
Conversation Memory - Maintains context across agent interactions.

KEY CONCEPT: Why Memory Matters for Agents
==========================================

Without memory, every request is independent:
- User: "What's 2 + 2?"
- Agent: "4"
- User: "Multiply that by 3"
- Agent: "Multiply what by 3?" ← No context!

With memory, agents can:
1. Continue conversations naturally
2. Reference previous tool results
3. Build on prior reasoning
4. Handle follow-up questions

MEMORY TYPES
============

1. Short-term (Conversation) Memory:
   - The current chat session
   - Stored in RAM
   - Lost when session ends
   
2. Long-term Memory:
   - Persisted across sessions
   - Stored in database/vector store
   - Enables "remember when we talked about X?"

3. Working Memory:
   - Current task context
   - Tool results being processed
   - Intermediate reasoning steps

This module implements conversation (short-term) memory.
For long-term memory, you'd typically use a vector database (like Week 2!).

MEMORY MANAGEMENT CHALLENGES
============================

1. Context Window Limits:
   - LLMs have max token limits (e.g., 128K for Claude)
   - Old messages must be summarized or removed
   
2. Relevance:
   - Not all history is equally important
   - Recent messages usually matter more
   
3. Cost:
   - Longer context = more expensive
   - Balance context vs. cost

Strategies:
- Sliding window: Keep last N messages
- Summarization: Compress old messages
- Selective: Keep important messages, drop filler
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from enum import Enum


class MessageRole(str, Enum):
    """Valid message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # Tool results


@dataclass
class Message:
    """
    A single message in the conversation.
    
    Attributes:
        role: Who sent the message (user, assistant, system, tool)
        content: The message text
        timestamp: When the message was created
        metadata: Additional info (tool name, token count, etc.)
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dict format for LLM API calls."""
        return {
            "role": self.role.value,
            "content": self.content
        }
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role.value}, content='{preview}')"


@dataclass
class ConversationMemory:
    """
    Manages conversation history for an agent session.
    
    Features:
    - Stores messages chronologically
    - Supports max message limits
    - Provides message filtering and formatting
    - Tracks token usage (approximate)
    
    Usage:
        memory = ConversationMemory(max_messages=50)
        memory.add_user_message("What's the weather?")
        memory.add_assistant_message("Let me check...")
        memory.add_tool_result("weather_tool", {"temp": 72})
        
        # Get messages for LLM
        messages = memory.get_messages()
    """
    max_messages: int = 100
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize with system prompt if provided."""
        if self.system_prompt:
            self.messages.append(Message(
                role=MessageRole.SYSTEM,
                content=self.system_prompt
            ))
    
    def add_message(self, role: MessageRole, content: str, **metadata) -> Message:
        """
        Add a message to memory.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message text
            **metadata: Additional metadata to store
            
        Returns:
            The created Message
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata
        )
        self.messages.append(message)
        
        # Enforce max messages (but keep system prompt)
        self._trim_if_needed()
        
        return message
    
    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        return self.add_message(MessageRole.USER, content)
    
    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content)
    
    def add_tool_result(self, tool_name: str, result: any) -> Message:
        """
        Add a tool result to memory.
        
        Tool results are formatted as user messages (how Claude expects them)
        but marked with tool metadata.
        """
        content = f"Tool '{tool_name}' returned: {result}"
        return self.add_message(
            MessageRole.USER,
            content,
            tool_name=tool_name,
            is_tool_result=True
        )
    
    def get_messages(self, include_system: bool = True) -> list[dict]:
        """
        Get messages formatted for LLM API calls.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = []
        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM and not include_system:
                continue
            messages.append(msg.to_dict())
        return messages
    
    def get_last_n_messages(self, n: int) -> list[dict]:
        """Get the last N messages (excluding system)."""
        non_system = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        return [m.to_dict() for m in non_system[-n:]]
    
    def get_user_messages(self) -> list[Message]:
        """Get only user messages."""
        return [m for m in self.messages if m.role == MessageRole.USER]
    
    def get_assistant_messages(self) -> list[Message]:
        """Get only assistant messages."""
        return [m for m in self.messages if m.role == MessageRole.ASSISTANT]
    
    def _trim_if_needed(self):
        """
        Trim old messages if we exceed max_messages.
        
        Strategy: Keep system message, remove oldest non-system messages.
        
        More sophisticated approaches:
        - Summarize old messages before removing
        - Use importance scoring
        - Keep messages with high engagement
        """
        if len(self.messages) <= self.max_messages:
            return
        
        # Separate system and non-system messages
        system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        other_msgs = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        
        # Keep max_messages - len(system_msgs) non-system messages
        keep_count = self.max_messages - len(system_msgs)
        trimmed = other_msgs[-keep_count:] if keep_count > 0 else []
        
        # Rebuild messages list
        self.messages = system_msgs + trimmed
    
    def clear(self, keep_system: bool = True):
        """
        Clear conversation history.
        
        Args:
            keep_system: Whether to preserve the system prompt
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        else:
            self.messages = []
    
    def get_summary(self) -> dict:
        """Get a summary of the conversation."""
        return {
            "total_messages": len(self.messages),
            "user_messages": len(self.get_user_messages()),
            "assistant_messages": len(self.get_assistant_messages()),
            "has_system_prompt": any(m.role == MessageRole.SYSTEM for m in self.messages),
        }
    
    def __len__(self) -> int:
        """Number of messages in memory."""
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self.messages)}, max={self.max_messages})"


class SessionManager:
    """
    Manages multiple conversation sessions.
    
    Useful for:
    - Multi-user applications
    - Separate conversations per topic
    - Session persistence
    
    Usage:
        manager = SessionManager()
        session1 = manager.get_or_create("user_123")
        session2 = manager.get_or_create("user_456")
    """
    
    def __init__(self, default_max_messages: int = 100):
        """Initialize session manager."""
        self._sessions: dict[str, ConversationMemory] = {}
        self._default_max_messages = default_max_messages
    
    def get_or_create(
        self,
        session_id: str,
        system_prompt: str | None = None
    ) -> ConversationMemory:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Unique session identifier
            system_prompt: System prompt for new sessions
            
        Returns:
            ConversationMemory for the session
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(
                max_messages=self._default_max_messages,
                system_prompt=system_prompt
            )
        return self._sessions[session_id]
    
    def get(self, session_id: str) -> ConversationMemory | None:
        """Get a session by ID, or None if not found."""
        return self._sessions.get(session_id)
    
    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return list(self._sessions.keys())
    
    def clear_all(self):
        """Delete all sessions."""
        self._sessions.clear()
