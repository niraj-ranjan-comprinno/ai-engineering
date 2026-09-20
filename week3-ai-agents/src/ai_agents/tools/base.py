"""
Base Tool class - The foundation for all agent tools.

KEY CONCEPT: Tools are how agents interact with the world
================================================

An LLM by itself can only generate text. To be useful, it needs to:
1. Search the web
2. Read files
3. Call APIs
4. Do calculations
5. Access databases

Tools give the LLM these capabilities. Each tool has:
- name: Unique identifier (e.g., "calculator")
- description: What it does (LLM reads this to decide when to use it)
- parameters: What inputs it needs (JSON Schema format)
- execute(): The actual logic that runs

The LLM doesn't run tools directly - it outputs a "tool call" (JSON)
that the agent framework interprets and executes.
"""
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Result returned after executing a tool.
    
    This is what the agent "observes" after taking an action.
    The observation gets fed back to the LLM for the next reasoning step.
    """
    success: bool = Field(description="Whether the tool executed successfully")
    result: Any = Field(description="The output/data from the tool")
    error: str | None = Field(default=None, description="Error message if failed")
    
    def to_observation(self) -> str:
        """Format result as observation text for the LLM."""
        if self.success:
            return f"Tool executed successfully. Result: {self.result}"
        else:
            return f"Tool failed with error: {self.error}"


class Tool(ABC):
    """
    Abstract base class for all tools.
    
    To create a new tool:
    1. Inherit from Tool
    2. Set name, description, and parameters
    3. Implement the execute() method
    
    Example:
        class WeatherTool(Tool):
            name = "get_weather"
            description = "Get current weather for a city"
            parameters = {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
            
            async def execute(self, city: str) -> ToolResult:
                # ... fetch weather data ...
                return ToolResult(success=True, result=weather_data)
    """
    
    # Subclasses must define these
    name: str  # Unique tool identifier
    description: str  # What the tool does (LLM reads this!)
    parameters: dict  # JSON Schema for inputs
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        This is where the actual work happens:
        - API calls
        - Calculations
        - File operations
        - Database queries
        
        Args:
            **kwargs: Parameters as defined in the tool's schema
            
        Returns:
            ToolResult with success/failure and data
        """
        pass
    
    def to_schema(self) -> dict:
        """
        Convert tool to function calling schema.
        
        This format is what Claude/GPT expect for function calling.
        The LLM uses this schema to:
        1. Know what tools are available
        2. Understand what each tool does
        3. Know what parameters to provide
        
        Returns:
            Dict in Claude's tool schema format
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }
    
    def __repr__(self) -> str:
        return f"Tool(name={self.name})"
