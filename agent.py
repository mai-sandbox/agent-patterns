"""
Simple chat agent with tool calling using LangGraph.

This agent implements a basic chatbot that can use tools (specifically Tavily search)
to answer questions beyond its training data. The system prompt is configurable
via the assistant configuration.
"""

from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langchain_tavily import TavilySearch
from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat agent."""
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent_graph() -> StateGraph:
    """Create and configure the agent graph."""
    
    # Initialize the search tool
    tool = TavilySearch(max_results=2)
    tools = [tool]
    
    # Initialize the LLM with tool binding
    # Using Anthropic Claude as the preferred model
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.1
    )  # type: ignore[call-arg]
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State) -> Dict[str, Any]:
        """
        Main chatbot node that processes messages and decides whether to use tools.
        
        Args:
            state: Current state containing message history
            
        Returns:
            Dictionary with updated messages
        """
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # Create the state graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    
    # Create tool node for executing tools
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.add_edge(START, "chatbot")
    
    # Add conditional edges for tool routing
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        # Route to tools if tool calls are present, otherwise end
        {"tools": "tools", END: END}
    )
    
    # After tool execution, return to chatbot
    graph_builder.add_edge("tools", "chatbot")
    
    return graph_builder


# Create and compile the graph
graph_builder = create_agent_graph()
graph = graph_builder.compile()

# Export the compiled graph as 'app' for LangGraph deployment compatibility
app = graph


