"""
Simple chat agent with tool calling using LangGraph.

This agent uses:
- StateGraph with MessagesState for conversation management
- Anthropic Claude as the LLM
- TavilySearch for web search capabilities
- Conditional edges for routing between chat and tool execution
"""

import os
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()


class State(TypedDict):
    """State schema for the chat agent."""
    messages: Annotated[list, add_messages]


def create_agent() -> StateGraph:
    """Create and configure the chat agent graph."""
    
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Initialize tools
    search_tool = TavilySearch(
        max_results=3,
        api_key=os.getenv("TAVILY_API_KEY")
    )
    tools = [search_tool]
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the state graph
    graph_builder = StateGraph(State)
    
    def chatbot(state: State) -> Dict[str, Any]:
        """Main chatbot node that processes messages and decides on tool usage."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # Add nodes to the graph
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
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Return to chatbot after tool execution
    graph_builder.add_edge("tools", "chatbot")
    
    return graph_builder


def main():
    """Main function for testing the agent locally."""
    # Create and compile the graph
    graph_builder = create_agent()
    graph = graph_builder.compile()
    
    # Test the agent
    test_input = {
        "messages": [
            {"role": "user", "content": "What's the weather like in San Francisco today?"}
        ]
    }
    
    print("Testing the agent...")
    for event in graph.stream(test_input):
        for value in event.values():
            if "messages" in value:
                last_message = value["messages"][-1]
                print(f"Agent: {last_message.content}")


# Create and compile the graph for deployment
graph_builder = create_agent()
graph = graph_builder.compile()

# MANDATORY: Export the compiled graph as 'app' for LangGraph deployment
app = graph


if __name__ == "__main__":
    main()
