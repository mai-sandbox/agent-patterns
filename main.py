"""Multi-functional web-enabled agent with weather, web search, and ice cream flavor capabilities."""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from tools import weather_tool, web_search_tool, ice_cream_flavor_tool


def setup_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check for required API keys
    required_keys = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY", "OPENWEATHERMAP_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print("⚠️  Warning: Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease copy .env.example to .env and fill in your API keys.")
        print("Some features may not work without proper API keys.\n")


def create_agent():
    """Create and configure the LangGraph agent with all tools."""
    try:
        # Initialize the language model
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            max_tokens=4000
        )
        
        # Define the tools list
        tools = [weather_tool, web_search_tool, ice_cream_flavor_tool]
        
        # Create the react agent
        agent = create_react_agent(llm, tools)
        
        return agent
    
    except Exception as e:
        print(f"❌ Error creating agent: {str(e)}")
        print("Please check your API keys and try again.")
        return None


def run_conversation(agent) -> None:
    """Run the conversational interface for the agent."""
    print("🤖 Multi-Functional Web-Enabled Agent")
    print("=" * 50)
    print("I can help you with:")
    print("• 🌤️  Weather information for any location")
    print("• 🔍 Web searches for general information")
    print("• 🍦 Current favorite ice cream flavor trends")
    print("\nType 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye! Thanks for using the Multi-Functional Agent!")
                break
            
            if not user_input:
                print("Please enter a question or request.")
                continue
            
            print("\n🤖 Agent: ", end="", flush=True)
            
            # Stream the agent's response
            for chunk in agent.stream({"messages": [{"role": "user", "content": user_input}]}):
                if "messages" in chunk and chunk["messages"]:
                    last_message = chunk["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        print(last_message.content)
        
        except KeyboardInterrupt:
