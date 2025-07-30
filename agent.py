"""
Simple LangGraph chat agent with tool calling capabilities.

This agent includes:
- A chatbot node with configurable system prompt
- Tavily search tool for web search capabilities
- Conditional edges for tool routing
- Proper state management using MessagesState
"""

from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat agent."""
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent_graph() -> StateGraph:
    """Create and configure the LangGraph agent."""
    
    # Initialize the language model
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.1,
    )
    
    # Initialize tools
    search_tool = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )
    tools = [search_tool]
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
        """
        Main chatbot node that processes messages and generates responses.
        
        The system prompt can be configured via the 'configurable' key in the config.
        """
        messages = state["messages"]
        
        # Get configurable system prompt from config
        configurable = config.get("configurable", {})
        system_prompt = configurable.get(
            "system_prompt", 
            "You are a helpful AI assistant. Use the search tool when you need current information or facts that you're not certain about."
        )
        
        # Prepend system message if not already present or if it needs updating
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        elif isinstance(messages[0], SystemMessage) and messages[0].content != system_prompt:
            # Update system message if it has changed
            messages = [SystemMessage(content=system_prompt)] + messages[1:]
        
        # Generate response
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    # Create the StateGraph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "chatbot")
    
    return graph_builder


# Create and compile the graph
graph_builder = create_agent_graph()
graph = graph_builder.compile()

# Export the compiled graph as 'app' (required for LangGraph deployment)
app = graph
