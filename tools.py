"""Custom tools for the multi-functional web-enabled agent."""

import os
from typing import Optional

from langchain_core.tools import tool
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_tavily import TavilySearchResults


@tool
def weather_tool(location: str) -> str:
    """Get current weather information for a specific location.
    
    Args:
        location: The location to get weather for (e.g., "London,GB" or "New York,US")
    
    Returns:
        Detailed weather information including temperature, humidity, wind, etc.
    """
    try:
        weather = OpenWeatherMapAPIWrapper()
        return weather.run(location)
    except Exception as e:
        return f"Error getting weather information: {str(e)}"


@tool
def web_search_tool(query: str, max_results: Optional[int] = 3) -> str:
    """Search the web for information using Tavily Search.
    
    Args:
        query: The search query to look for
        max_results: Maximum number of search results to return (default: 3)
    
    Returns:
        Search results with URLs, titles, and content snippets
    """
    try:
        search = TavilySearchResults(max_results=max_results)
        results = search.invoke({"query": query})
        
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content")
            formatted_results.append(f"{i}. {title}\n   URL: {url}\n   Content: {content}\n")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def ice_cream_flavor_tool() -> str:
    """Find information about the world's current favorite ice cream flavors.
    
    Returns:
        Information about popular ice cream flavors and current trends
    """
    try:
        search = TavilySearchResults(max_results=3)
        query = "most popular ice cream flavors 2024 favorite trends survey"
