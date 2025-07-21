"""
LangGraph Chat Agent with Tool Calling

This module defines a simple chat agent with web search capabilities using
StateGraph, ToolNode, and tools_condition. The system prompt is configurable.
"""

from typing import Annotated
from langchain_tavily import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat agent."""
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent(config: dict) -> StateGraph:
    """
    Create a chat agent with configurable system prompt and tool calling.
    
    Args:
        config: Configuration dictionary containing system_prompt
        
    Returns:
        Compiled StateGraph representing the chat agent
    """
    # Get system prompt from config
    system_prompt = config.get("system_prompt", 
        "You are a helpful AI assistant with access to web search. Use tools when needed to provide accurate and up-to-date information.")
    
    # Initialize LLM with system prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Initialize search tool
    search_tool = TavilySearchResults(max_results=3)
    tools = [search_tool]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State):
        """Main chatbot function that processes messages."""
        # Add system prompt as first message if not present
        messages = state["messages"]
        if not messages or messages[0].content != system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Create StateGraph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Return to chatbot after tool execution
    graph_builder.add_edge("tools", "chatbot")
    
    return graph_builder.compile()


# Create the graph instance that will be used by LangGraph server
def graph(config: dict = None):
    """Entry point for LangGraph server."""
    if config is None:
        config = {}
    return create_agent(config)

