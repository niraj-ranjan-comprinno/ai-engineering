"""
DateTime Tool - Get current date and time information.

WHY THIS TOOL EXISTS:
====================
LLMs don't know what time it is! Their training data has a cutoff,
and they have no concept of "now". This tool provides real-time
date/time information.

Common use cases:
- "What day is it today?"
- "How many days until Christmas?"
- "What time is it in Tokyo?"
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from .base import Tool, ToolResult


class DateTimeTool(Tool):
    """
    DateTime tool - get current date/time or timezone conversions.
    """
    
    name = "get_datetime"
    description = """Get current date and time information.
Use this when the user asks about:
- Current date or time
- Day of the week
- Time in different timezones
Optionally specify a timezone like 'America/New_York', 'Europe/London', 'Asia/Tokyo'."""
    
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name (e.g., 'America/New_York', 'UTC'). Defaults to UTC if not specified."
            }
        },
        "required": []
    }
    
    async def execute(self, timezone: str = "UTC") -> ToolResult:
        """Get current datetime for specified timezone."""
        try:
            # Get timezone
            if timezone.upper() == "UTC":
                tz = ZoneInfo("UTC")
            else:
                tz = ZoneInfo(timezone)
            
            # Get current time
            now = datetime.now(tz)
            
            result = {
                "timezone": timezone,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p"),
                "iso_format": now.isoformat(),
            }
            
            return ToolResult(success=True, result=result)
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Invalid timezone or error: {str(e)}"
            )
