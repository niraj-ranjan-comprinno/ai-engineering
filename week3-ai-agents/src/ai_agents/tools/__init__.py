"""Tools module - Tool definitions and registry."""
from .base import Tool, ToolResult
from .registry import ToolRegistry, default_registry
from .calculator import CalculatorTool
from .weather import WeatherTool
from .datetime_tool import DateTimeTool
from .web_search import WebSearchTool

__all__ = [
    "Tool",
    "ToolResult", 
    "ToolRegistry",
    "default_registry",
    "CalculatorTool",
    "WeatherTool",
    "DateTimeTool",
    "WebSearchTool",
]


def register_default_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """
    Register all built-in tools to a registry.
    
    Args:
        registry: Registry to use. If None, uses default_registry.
        
    Returns:
        The registry with tools registered.
    """
    if registry is None:
        registry = default_registry
    
    # Register all tools
    tools = [
        CalculatorTool(),
        WeatherTool(),
        DateTimeTool(),
        WebSearchTool(),
    ]
    
    for tool in tools:
        if tool.name not in registry:
            registry.register(tool)
    
    return registry
