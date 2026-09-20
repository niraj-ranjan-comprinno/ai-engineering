"""
Calculator Tool - Perform mathematical calculations.

WHY THIS TOOL EXISTS:
====================
LLMs are notoriously bad at math! They can understand math concepts
but often make arithmetic errors. A calculator tool lets the LLM
delegate calculations to reliable code.

Example:
- User: "What's 23.5 * 17.8 + 42?"
- Without tool: LLM might say "459.3" (wrong!)
- With tool: LLM calls calculator → gets exact answer "460.3"
"""
import math
from .base import Tool, ToolResult


class CalculatorTool(Tool):
    """
    Calculator tool for mathematical operations.
    
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, abs
    
    The LLM sends a math expression as a string,
    we safely evaluate it and return the result.
    """
    
    name = "calculator"
    description = """Perform mathematical calculations. 
Use this tool when you need to do arithmetic, solve equations, or compute values.
Input should be a valid mathematical expression like '2 + 2' or 'sqrt(16) * 3'.
Supported operations: +, -, *, /, ** (power), sqrt(), sin(), cos(), tan(), log(), abs()"""
    
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'"
            }
        },
        "required": ["expression"]
    }
    
    # Safe functions we allow in expressions
    SAFE_FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    
    async def execute(self, expression: str) -> ToolResult:
        """
        Safely evaluate a mathematical expression.
        
        We use a restricted eval() with only math functions allowed.
        This prevents code injection attacks.
        """
        try:
            # Clean up the expression
            expression = expression.strip()
            
            # Create safe evaluation context
            # Only allow math operations, no builtins
            safe_dict = {"__builtins__": {}}
            safe_dict.update(self.SAFE_FUNCTIONS)
            
            # Evaluate the expression
            result = eval(expression, safe_dict)
            
            # Round if it's a float with many decimals
            if isinstance(result, float):
                result = round(result, 10)
            
            return ToolResult(
                success=True,
                result=result
            )
            
        except ZeroDivisionError:
            return ToolResult(
                success=False,
                result=None,
                error="Division by zero"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Invalid expression: {str(e)}"
            )
