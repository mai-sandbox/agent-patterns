"""
LangGraph Agent with configurable system prompt and tool calling.
"""
from typing import List
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver


def get_weather(city: str) -> str:
    """Get weather information for a given city."""
    # Simple mock weather tool
    weather_data = {
        "san francisco": "Sunny, 72°F",
        "new york": "Cloudy, 65°F", 
        "london": "Rainy, 58°F",
        "tokyo": "Clear, 75°F"
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"Weather data not available for {city}. It's probably nice though!")


def search_web(query: str) -> str:
    """Search the web for information."""
    # Simple mock search tool
    return f"Here are some search results for '{query}': This is a mock search result. In a real implementation, this would connect to a search API."


def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # Simple calculator - be careful with eval in production!
        result = eval(expression.replace("^", "**"))
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def dynamic_prompt(state: AgentState, config: RunnableConfig) -> List[AnyMessage]:
    """
    Dynamic prompt function that creates system message based on configuration.
    This allows the system prompt to be updated at runtime.
    """
    # Get the system prompt from config, with a default fallback
    system_prompt = config.get("configurable", {}).get(
        "system_prompt", 
        "You are a helpful AI assistant with access to tools. Use them when appropriate to help users."
    )
    
    # Create system message and add existing conversation messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["messages"])
    
    return messages


# Define the tools available to the agent
tools = [get_weather, search_web, calculate]

# Create the model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create checkpointer for memory
checkpointer = InMemorySaver()

# Create the agent with dynamic prompt
graph = create_react_agent(
    model=model,
    tools=tools,
    prompt=dynamic_prompt,  # Use our dynamic prompt function
    checkpointer=checkpointer
)
