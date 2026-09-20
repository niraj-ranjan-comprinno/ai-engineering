"""
Web Search Tool - Search the internet for information.

WHY THIS TOOL EXISTS:
====================
LLMs have knowledge cutoffs and can't access the internet.
A search tool lets the agent find current information:
- Latest news
- Product prices
- Company information
- Technical documentation

NOTE: This is a MOCK implementation for learning.
In production, you'd use:
- Google Custom Search API
- Bing Search API
- SerpAPI
- Tavily (popular for AI agents)
"""
from .base import Tool, ToolResult


class WebSearchTool(Tool):
    """
    Web search tool - search the internet for information.
    
    Mock implementation that returns simulated search results.
    Replace with real search API for production use.
    """
    
    name = "web_search"
    description = """Search the web for current information.
Use this when you need to find:
- Current news or events
- Product information or prices
- Company details
- Technical documentation
- Any information that might have changed since your training"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 3, max: 10)"
            }
        },
        "required": ["query"]
    }
    
    # Mock search results for demo
    MOCK_RESULTS = {
        "python": [
            {
                "title": "Python.org - Official Website",
                "url": "https://python.org",
                "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively."
            },
            {
                "title": "Python Tutorial - W3Schools",
                "url": "https://w3schools.com/python",
                "snippet": "Learn Python programming with our comprehensive tutorial covering basics to advanced topics."
            }
        ],
        "weather": [
            {
                "title": "Weather.com - Local Forecast",
                "url": "https://weather.com",
                "snippet": "Get accurate local weather forecasts, radar maps, and severe weather alerts."
            }
        ],
        "ai": [
            {
                "title": "What is Artificial Intelligence (AI)?",
                "url": "https://example.com/ai",
                "snippet": "Artificial Intelligence refers to systems that can perform tasks requiring human intelligence."
            },
            {
                "title": "Latest AI News and Developments",
                "url": "https://example.com/ai-news",
                "snippet": "Stay updated with the latest breakthroughs in artificial intelligence and machine learning."
            }
        ]
    }
    
    async def execute(self, query: str, num_results: int = 3) -> ToolResult:
        """
        Search the web (mock implementation).
        
        In production:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.com/search",
                    params={"q": query, "num": num_results},
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                return ToolResult(success=True, result=response.json())
        """
        try:
            num_results = min(max(1, num_results), 10)
            
            # Find matching mock results
            results = []
            query_lower = query.lower()
            
            for keyword, keyword_results in self.MOCK_RESULTS.items():
                if keyword in query_lower:
                    results.extend(keyword_results)
            
            # If no matches, return generic results
            if not results:
                results = [
                    {
                        "title": f"Search results for: {query}",
                        "url": f"https://search.example.com?q={query}",
                        "snippet": f"Found information related to '{query}'. This is simulated search data for demonstration purposes."
                    }
                ]
            
            return ToolResult(
                success=True,
                result={
                    "query": query,
                    "num_results": len(results[:num_results]),
                    "results": results[:num_results]
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Search failed: {str(e)}"
            )
