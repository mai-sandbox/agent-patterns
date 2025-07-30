"""
Simple chat agent with tool calling capability and configurable system prompt.
This agent uses StateGraph with add_messages for conversation state management.
"""

from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat agent."""
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent_graph():
    """Create and return the compiled agent graph."""
    
    # Initialize the LLM (using Anthropic as preferred)
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.7
    )
    
    # Initialize tools
    search_tool = TavilySearch(
        max_results=3,
        description="Search the web for current information"
    )
    tools = [search_tool]
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
        """
        Main agent node that processes messages and calls the LLM.
        Uses configurable system prompt from the runtime configuration.
        """
        messages = state["messages"]
        
        # Get system prompt from configuration (default if not provided)
        configurable = config.get("configurable", {})
        system_prompt = configurable.get(
            "system_prompt", 
            "You are a helpful AI assistant with access to web search. "
            "Use the search tool when you need current information or facts "
            "that might not be in your training data."
        )
        
        # Add system message if not already present or if it's different
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        elif messages[0].content != system_prompt:
            # Update system message if it's different
            messages = [SystemMessage(content=system_prompt)] + messages[1:]
        
        # Call the LLM with tools
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    # Create the StateGraph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "agent")
    
    # Compile the graph
    return graph_builder.compile()


# Create and export the compiled graph as 'app' (required for deployment)
app = create_agent_graph()

if __name__ == "__main__":
    # Test the agent locally
    import asyncio
    
    async def test_agent():
        """Test the agent with a simple query."""
        config = {
            "configurable": {
                "system_prompt": "You are a helpful assistant that loves to help with questions."
            }
        }
        
        result = await app.ainvoke(
            {"messages": [{"role": "user", "content": "What is LangGraph?"}]},
            config=config
        )
        
        print("Agent response:")
        print(result["messages"][-1].content)
    
    # Run the test
    asyncio.run(test_agent())
