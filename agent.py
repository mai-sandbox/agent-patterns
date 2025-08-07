"""
Simple chat agent with tool calling capabilities for LangGraph.
Includes configurable system prompt and basic tools.
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


# Define the state schema using TypedDict
class AgentState(TypedDict):
    """State schema for the chat agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    system_prompt: str


# Define basic tools
@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Only allow basic mathematical operations for safety
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def web_search_simulator(query: str) -> str:
    """
    Simulate a web search (placeholder implementation).
    
    Args:
        query: The search query
    
    Returns:
        A simulated search result
    """
    return f"Simulated search results for '{query}': This is a placeholder web search tool. In a real implementation, this would connect to a search API like Google, Bing, or DuckDuckGo to provide actual search results."


# Create the tools list
tools = [calculator, web_search_simulator]


def create_agent_with_config(system_prompt: str = "You are a helpful AI assistant with access to tools."):
    """
    Create a LangGraph agent with the specified system prompt.
    
    Args:
        system_prompt: The system prompt to use for the agent
    
    Returns:
        A compiled LangGraph StateGraph
    """
    
    # Initialize the language model
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    model_with_tools = model.bind_tools(tools)
    
    def should_continue(state: AgentState) -> str:
        """Determine whether to continue or end the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes a tool call, then we route to the "tools" node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise, we stop (reply to the user)
        return END
    
    def call_model(state: AgentState) -> dict:
        """Call the language model with the current state."""
        messages = state["messages"]
        system_prompt = state.get("system_prompt", "You are a helpful AI assistant with access to tools.")
        
        # Add system message if not already present or if it needs updating
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        elif isinstance(messages[0], SystemMessage) and messages[0].content != system_prompt:
            # Update system message if it has changed
            messages = [SystemMessage(content=system_prompt)] + list(messages[1:])
        
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set the entrypoint
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# Default agent instance
def get_default_agent():
    """Get the default agent instance."""
    return create_agent_with_config()


# For LangGraph deployment
app = get_default_agent()


if __name__ == "__main__":
    # Test the agent locally
    agent = get_default_agent()
    
    # Test with a simple message
    initial_state = {
        "messages": [HumanMessage(content="Hello! Can you calculate 15 * 7 for me?")],
        "system_prompt": "You are a helpful AI assistant with access to tools."
    }
    
    print("Testing the agent...")
    result = agent.invoke(initial_state)
    
    print("\nConversation:")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, SystemMessage):
            print(f"System: {message.content}")
        else:
            print(f"Assistant: {message.content}")


