"""
LangGraph Chat Agent with Tool Calling

This module implements a simple chat agent with tool calling capabilities using LangGraph's
create_react_agent. The agent includes basic tools like weather and calculator, with a
configurable system prompt parameter.
"""

import os
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph


# Define basic tools for the agent
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location.

    Args:
        location: The city or location to get weather for

    Returns:
        A string describing the current weather conditions
    """
    # Simulate weather data - in a real implementation, you'd call a weather API
    weather_data = {
        "new york": "Sunny, 72°F with light winds",
        "london": "Cloudy, 15°C with occasional rain",
        "tokyo": "Partly cloudy, 25°C with high humidity",
        "paris": "Clear skies, 18°C with gentle breeze",
        "sydney": "Overcast, 22°C with strong winds",
    }

    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    else:
        return f"Weather in {location}: Partly cloudy, 20°C (simulated data - location not in database)"


@tool
def calculator(expression: str) -> str:
    """Perform basic mathematical calculations.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        The result of the calculation as a string
    """
    try:
        # Use eval safely for basic math operations
        # In production, you'd want to use a more secure math parser
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."

        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def get_time() -> str:
    """Get the current time and date.

    Returns:
        Current date and time as a string
    """
    from datetime import datetime

    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# Define the tools list
tools = [get_weather, calculator, get_time]


def create_agent(system_prompt: Optional[str] = None) -> CompiledStateGraph:
    """Create a LangGraph chat agent with configurable system prompt.

    Args:
        system_prompt: Custom system prompt for the agent. If None, uses default.

    Returns:
        Compiled LangGraph agent ready for deployment
    """
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """You are a helpful AI assistant with access to tools. 
        You can help users with weather information, mathematical calculations, and provide the current time.
        
        When using tools:
        - For weather: Ask for a specific location if not provided
        - For calculations: Ensure the expression is mathematically valid
        - Always provide clear, helpful responses
        
        Be conversational and friendly while being accurate and helpful."""

    # Initialize the language model (preferring Anthropic as per guidelines)
    model = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022", temperature=0.1
    )

    # Create the react agent with tools and system prompt
    agent = create_react_agent(model=model, tools=tools, prompt=system_prompt)

    return agent


# Initialize the language model (preferring Anthropic as per guidelines)
model = ChatAnthropic(
    model_name="claude-3-5-sonnet-20241022", temperature=0.1, max_tokens=2048
)

# Default system prompt
default_system_prompt = """You are a helpful AI assistant with access to tools. 
You can help users with weather information, mathematical calculations, and provide the current time.

When using tools:
- For weather: Ask for a specific location if not provided
- For calculations: Ensure the expression is mathematically valid
- Always provide clear, helpful responses

Be conversational and friendly while being accurate and helpful."""

# Create the react agent with tools and default system prompt
# The system prompt can be overridden via configuration in langgraph.json
app = create_react_agent(model=model, tools=tools, prompt=default_system_prompt)


if __name__ == "__main__":
    # Test the agent locally
    print("Testing LangGraph Chat Agent...")

    # Test with default system prompt
    test_agent = create_agent()

    # Example test messages
    test_messages = [
        {"role": "human", "content": "What's the weather like in New York?"},
        {"role": "human", "content": "Calculate 15 * 7 + 23"},
        {"role": "human", "content": "What time is it?"},
    ]

    for msg in test_messages:
        print(f"\nUser: {msg['content']}")
        try:
            result = test_agent.invoke({"messages": [msg]})
            if result.get("messages"):
                assistant_msg = result["messages"][-1]
                print(f"Assistant: {assistant_msg.content}")
        except Exception as e:
            print(f"Error: {e}")

