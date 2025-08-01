"""
Custom tools for web search and webpage summarization.

This module provides two main tools:
1. Tavily web search tool for general web searches
2. Firecrawl webpage summarization tool for extracting and summarizing webpage content
"""

import json
from typing import Dict, Any, Optional

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders.firecrawl import FireCrawlLoader


@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily Search Engine.
    
    This tool performs web searches and returns relevant results with summaries,
    URLs, and content snippets. Useful for finding current information, news,
    and general web content.
    
    Args:
        query: The search query string to find relevant web content
        
    Returns:
        A formatted string containing search results with titles, URLs, and content snippets
        
    Raises:
        Exception: If the search fails due to API issues or invalid query
    """
    try:
        # Initialize Tavily search with max 2 results for focused responses
        search_tool = TavilySearch(max_results=2)
        
        # Perform the search
        search_results = search_tool.invoke(query)
        
        if not search_results or 'results' not in search_results:
            return f"No search results found for query: {query}"
        
        # Format the results for better readability
        formatted_results = []
        formatted_results.append(f"**Web Search Results for: '{query}'**\n")
        
        for i, result in enumerate(search_results['results'], 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No content available')
            
            formatted_results.append(f"**Result {i}:**")
            formatted_results.append(f"Title: {title}")
            formatted_results.append(f"URL: {url}")
            formatted_results.append(f"Summary: {content[:500]}{'...' if len(content) > 500 else ''}")
            formatted_results.append("")  # Empty line for separation
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Web search failed for query '{query}': {str(e)}"
        return error_msg


@tool
def summarize_webpage(url: str) -> str:
    """
    Scrape and summarize a webpage using Firecrawl.
    
    This tool uses Firecrawl to extract clean markdown content from a webpage
    and returns it for summarization. It handles dynamic content, JavaScript-rendered
    pages, and provides clean, LLM-ready markdown output.
    
    Args:
        url: The URL of the webpage to scrape and summarize
        
    Returns:
        Clean markdown content from the webpage, ready for summarization
        
    Raises:
        Exception: If the webpage scraping fails due to network issues, invalid URL,
                  or Firecrawl API problems
    """
    try:
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return f"Invalid URL format: {url}. Please provide a complete URL starting with http:// or https://"
        
        # Initialize Firecrawl loader in scrape mode
        loader = FireCrawlLoader(
            api_key=None,  # Will use FIRECRAWL_API_KEY from environment
            url=url,
            mode="scrape"
        )
        
        # Load the webpage content
        documents = loader.load()
        
        if not documents:
            return f"No content could be extracted from URL: {url}"
        
        # Get the first document (single page scrape)
        doc = documents[0]
        
        # Extract metadata for context
        metadata = doc.metadata
        title = metadata.get('title', 'Unknown Title')
        description = metadata.get('description', 'No description available')
        
        # Format the response with metadata and content
        result_parts = []
        result_parts.append(f"**Webpage Content Summary**")
        result_parts.append(f"**URL:** {url}")
        result_parts.append(f"**Title:** {title}")
        result_parts.append(f"**Description:** {description}")
        result_parts.append("")
        result_parts.append("**Content:**")
        result_parts.append(doc.page_content)
        
        return "\n".join(result_parts)
        
    except Exception as e:
        error_msg = f"Failed to scrape webpage '{url}': {str(e)}"
        return error_msg


# Export the tools for use in the agent
tools = [web_search, summarize_webpage]
