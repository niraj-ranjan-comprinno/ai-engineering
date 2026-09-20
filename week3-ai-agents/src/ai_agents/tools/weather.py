"""
Weather Tool - Get current weather for a location.

WHY THIS TOOL EXISTS:
====================
LLMs don't have access to real-time data. Their knowledge is frozen
at training time. A weather tool lets the agent access live information.

NOTE: This is a MOCK implementation for learning purposes.
In production, you'd call a real weather API like:
- OpenWeatherMap
- WeatherAPI
- AccuWeather

The important thing is understanding the PATTERN:
1. LLM decides it needs weather data
2. Calls the weather tool with a city name
3. Tool fetches real data from API
4. Returns data for LLM to use in response
"""
import random
from .base import Tool, ToolResult


class WeatherTool(Tool):
    """
    Weather tool - returns current weather for a city.
    
    This is a mock implementation that returns simulated data.
    In production, replace with real API calls.
    """
    
    name = "get_weather"
    description = """Get current weather conditions for a city.
Use this when the user asks about weather, temperature, or conditions in a location.
Returns temperature, conditions (sunny, cloudy, rainy, etc.), and humidity."""
    
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g., 'New York', 'London', 'Tokyo'"
            }
        },
        "required": ["city"]
    }
    
    # Mock weather data for demo purposes
    CONDITIONS = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy"]
    
    async def execute(self, city: str) -> ToolResult:
        """
        Get weather for a city (mock implementation).
        
        In production, this would call a real weather API:
        
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.weather.com/v1/current",
                    params={"city": city, "apikey": API_KEY}
                )
                return ToolResult(success=True, result=response.json())
        """
        try:
            # Simulate weather data
            # In production: call real API here
            weather_data = {
                "city": city.title(),
                "temperature_celsius": random.randint(5, 35),
                "temperature_fahrenheit": None,  # Will calculate
                "conditions": random.choice(self.CONDITIONS),
                "humidity_percent": random.randint(30, 90),
                "wind_speed_kmh": random.randint(5, 40),
            }
            
            # Calculate Fahrenheit
            weather_data["temperature_fahrenheit"] = round(
                weather_data["temperature_celsius"] * 9/5 + 32
            )
            
            return ToolResult(
                success=True,
                result=weather_data
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Failed to get weather: {str(e)}"
            )
