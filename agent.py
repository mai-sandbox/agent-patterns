"""
Web-enabled AI Agent with Web Search and Webpage Summarization

This module implements a LangGraph-based agent that can:
1. Perform web searches using Tavily Search Engine
2. Summarize webpages using Firecrawl
3. Engage in conversational interactions with users

The agent uses a StateGraph architecture with conditional routing to determine
when to use tools versus providing direct responses.
"""

from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools

# Load environment variables
load_dotenv()


class State(TypedDict):
    """
    State schema for the agent conversation.
    
    The state contains a list of messages that represents the conversation history.
    The add_messages annotation ensures proper message handling and state updates.
    """
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent():
    """
    Create and configure the LangGraph agent with web search and webpage summarization capabilities.
    
    Returns:
        Compiled graph ready for execution
    """
    # Initialize the language model (Anthropic Claude as preferred)
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.1
    )
    
    # Bind tools to the LLM so it knows what tools are available
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State) -> dict:
        """
        Main chatbot node that processes user messages and decides whether to use tools.
        
        Args:
            state: Current conversation state containing messages
            
        Returns:
            Dictionary with updated messages containing the LLM response
        """
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    # Create the StateGraph
    graph_builder = StateGraph(State)
    
    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot)
    
    # Add the tool execution node using prebuilt ToolNode
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Define the conversation flow with conditional edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,  # Prebuilt condition that checks for tool calls
        {
            "tools": "tools",  # If tools are called, go to tools node
            END: END  # If no tools, end the conversation turn
        }
    )
    
    # After tools are executed, return to chatbot for response
    graph_builder.add_edge("tools", "chatbot")
    
    # Start the conversation with the chatbot
    graph_builder.add_edge(START, "chatbot")
    
    # Compile the graph
    return graph_builder.compile()


# Create the agent graph
graph = create_agent()

# Export as 'app' for LangGraph server compatibility
app = graph


def main():
    """
    Main function for testing the agent locally.
    
    Provides a simple command-line interface to interact with the agent.
    """
    print("🔥 Web-Enabled AI Agent")
    print("=" * 50)
    print("I can help you with:")
    print("• Web searches using Tavily")
    print("• Webpage summarization using Firecrawl")
    print("• General conversation and questions")
    print("\nType 'quit', 'exit', or 'q' to stop.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                print("Please enter a message.")
                continue
            
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Stream the agent's response
            for event in app.stream({"messages": [{"role": "user", "content": user_input}]}):
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        last_message = value["messages"][-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            print(last_message.content)
                        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()

