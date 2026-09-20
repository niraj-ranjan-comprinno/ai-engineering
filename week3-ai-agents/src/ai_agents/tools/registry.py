"""
Tool Registry - Central place to register and discover tools.

KEY CONCEPT: Why a Registry?
============================

Agents need to know what tools are available. The registry:
1. Stores all registered tools
2. Provides lookup by name (for execution)
3. Generates tool schemas for the LLM
4. Enables dynamic tool discovery

Think of it like a plugin system - tools can be added/removed
without changing the agent code.

PATTERN: Registry Pattern
========================
Common in plugin architectures:
- Tools register themselves
- Agent queries registry for available tools
- Easy to extend with new capabilities
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Tool


class ToolRegistry:
    """
    Central registry for managing available tools.
    
    Usage:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WeatherTool())
        
        # Get tool for execution
        calc = registry.get("calculator")
        
        # Get all schemas for LLM
        schemas = registry.get_all_schemas()
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, "Tool"] = {}
    
    def register(self, tool: "Tool") -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        print(f"📦 Registered tool: {tool.name}")
    
    def get(self, name: str) -> "Tool | None":
        """
        Get a tool by name.
        
        Args:
            name: Tool name to look up
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_all(self) -> list["Tool"]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_all_schemas(self) -> list[dict]:
        """
        Get tool schemas for all registered tools.
        
        This is what we send to the LLM so it knows
        what tools are available and how to use them.
        
        Returns:
            List of tool schemas in Claude's format
        """
        return [tool.to_schema() for tool in self._tools.values()]
    
    def list_tools(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools


# Global default registry (singleton pattern)
# Tools can register here for easy access
default_registry = ToolRegistry()
