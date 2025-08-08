#!/usr/bin/env python3
"""
LangGraph-based Chatbot Implementation

A comprehensive chatbot built with LangGraph that supports multiple LLM providers,
proper state management, and interactive command-line interface.
"""

import os
import sys
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Load environment variables from .env file
load_dotenv()


class State(TypedDict):
    """
    State schema for the chatbot.
    
    The messages list uses the add_messages reducer to append new messages
    rather than overwriting the existing list.
    """
    messages: Annotated[list[BaseMessage], add_messages]


class ChatbotError(Exception):
    """Custom exception for chatbot-related errors."""
    pass


class LLMProvider:
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    
    @classmethod
    def get_all_providers(cls) -> list[str]:
        """Get list of all supported providers."""
        return [cls.OPENAI, cls.ANTHROPIC, cls.GOOGLE]


def get_available_provider() -> Optional[str]:
    """
    Determine which LLM provider is available based on environment variables.
    
    Returns:
        The name of the first available provider, or None if none are configured.
    """
    provider_configs = {
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY", 
        LLMProvider.GOOGLE: "GOOGLE_API_KEY"
    }
    
    for provider, env_var in provider_configs.items():
        if os.getenv(env_var):
            return provider
    
    return None


def initialize_llm(provider: Optional[str] = None) -> Any:
    """
    Initialize the language model based on the specified or available provider.
    
    Args:
        provider: The LLM provider to use. If None, auto-detect from environment.
        
    Returns:
        Initialized language model instance.
        
    Raises:
        ChatbotError: If no provider is available or initialization fails.
    """
    if provider is None:
        provider = get_available_provider()
    
    if provider is None:
        raise ChatbotError(
            "No LLM provider configured. Please set one of the following environment variables:\n"
            "- OPENAI_API_KEY for OpenAI\n"
            "- ANTHROPIC_API_KEY for Anthropic\n"
            "- GOOGLE_API_KEY for Google Gemini"
        )
    
    try:
        if provider == LLMProvider.OPENAI:
            return init_chat_model("openai:gpt-4o-mini")
        elif provider == LLMProvider.ANTHROPIC:
            return init_chat_model("anthropic:claude-3-5-sonnet-latest")
        elif provider == LLMProvider.GOOGLE:
            return init_chat_model("google_genai:gemini-2.0-flash")
        else:
            raise ChatbotError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        raise ChatbotError(f"Failed to initialize {provider} LLM: {str(e)}")


def chatbot_node(state: State) -> Dict[str, Any]:
    """
    The main chatbot node that processes messages and generates responses.
    
    Args:
        state: Current state containing the conversation messages.
        
    Returns:
        Dictionary containing the updated messages list.
        
    Raises:
        ChatbotError: If LLM invocation fails.
    """
    try:
        llm = initialize_llm()
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        raise ChatbotError(f"Error generating response: {str(e)}")


def create_chatbot_graph() -> Any:
    """
    Create and compile the LangGraph chatbot workflow.
    
    Returns:
        Compiled LangGraph instance ready for execution.
    """
    # Create the state graph
    graph_builder = StateGraph(State)
    
    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot_node)
    
    # Define the flow: START -> chatbot -> END
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    
    # Compile the graph
    return graph_builder.compile()


def stream_graph_updates(graph: Any, user_input: str) -> None:
    """
    Stream updates from the graph execution and display the assistant's response.
    
    Args:
        graph: Compiled LangGraph instance.
        user_input: User's input message.
    """
    try:
        # Create the initial state with user message
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Stream the graph execution
        for event in graph.stream(initial_state):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    # Get the last message (assistant's response)
                    last_message = value["messages"][-1]
                    print(f"Assistant: {last_message.content}")
                    
    except ChatbotError as e:
        print(f"Chatbot Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def display_welcome_message() -> None:
    """Display welcome message and usage instructions."""
    provider = get_available_provider()
    provider_name = provider.title() if provider else "Unknown"
    
    print("=" * 60)
    print("🤖 LangGraph Chatbot")
    print("=" * 60)
    print(f"Using LLM Provider: {provider_name}")
    print("\nInstructions:")
    print("- Type your message and press Enter to chat")
    print("- Type 'quit', 'exit', or 'q' to end the conversation")
    print("- The chatbot will remember the conversation context")
    print("=" * 60)
    print()


def main() -> None:
    """
    Main function to run the interactive chatbot.
    """
    try:
        # Display welcome message
        display_welcome_message()
        
        # Create the chatbot graph
        print("Initializing chatbot...")
        graph = create_chatbot_graph()
        print("Chatbot ready! Start chatting below.\n")
        
        # Interactive chat loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye! Thanks for chatting!")
                    break
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Process the user input and stream response
                stream_graph_updates(graph, user_input)
                print()  # Add spacing between exchanges
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                break
            except EOFError:
                # Handle case where input() is not available (e.g., in some environments)
                print("Input not available. Running demo conversation...")
                demo_input = "What is LangGraph and how does it work?"
                print(f"You: {demo_input}")
                stream_graph_updates(graph, demo_input)
                break
                
    except ChatbotError as e:
        print(f"❌ Chatbot Error: {e}")
        print("\nPlease check your configuration and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
