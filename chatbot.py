#!/usr/bin/env python3
"""
Basic Chatbot Implementation using LangGraph StateGraph.

This module implements a conversational chatbot using LangGraph's StateGraph
with Anthropic Claude as the LLM backend. It includes proper state management,
error handling, and logging following LangGraph best practices.
"""

import logging
import os
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log") if os.getenv("DEBUG_MODE", "false").lower() == "true" else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """
    State schema for the chatbot conversation.
    
    Attributes:
        messages: List of conversation messages with automatic message addition
        user_id: Optional identifier for the user
        session_id: Unique session identifier
        created_at: Timestamp when the conversation started
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: Optional[str]
    session_id: str
    created_at: str


class ChatbotError(Exception):
    """Custom exception for chatbot-related errors."""
    pass


class BasicChatbot:
    """
    Basic chatbot implementation using LangGraph StateGraph.
    
    This chatbot maintains conversation state and uses Anthropic Claude
    for generating responses. It includes comprehensive error handling
    and logging capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize the chatbot with configuration from environment variables."""
        try:
            # Validate required environment variables
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key or api_key == "your_anthropic_api_key_here":
                raise ChatbotError(
                    "ANTHROPIC_API_KEY not found or not set. "
                    "Please set your API key in the .env file."
                )
            
            # Initialize LLM with configuration
            self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
            self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
            self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
            
            self.llm = ChatAnthropic(
                model_name=self.model_name,
                max_tokens_to_sample=self.max_tokens,
                temperature=self.temperature,
                anthropic_api_key=api_key
            )
            
            # Build the conversation graph
            self.graph = self._build_graph()
            
            logger.info(
                f"Chatbot initialized with model: {self.model_name}, "
                f"max_tokens: {self.max_tokens}, temperature: {self.temperature}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise ChatbotError(f"Initialization failed: {e}") from e
    
    def _build_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph StateGraph for conversation flow.
        
        Returns:
            StateGraph: Configured conversation graph
        """
        try:
            # Create the state graph
            workflow = StateGraph(ChatState)
            
            # Add nodes
            workflow.add_node("llm_node", self._llm_node)
            
            # Set entry point
            workflow.set_entry_point("llm_node")
            
            # Add edges
            workflow.add_edge("llm_node", END)
            
            # Compile the graph
            graph = workflow.compile()
            
            logger.debug("Conversation graph built successfully")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build conversation graph: {e}")
            raise ChatbotError(f"Graph building failed: {e}") from e
    
    def _llm_node(self, state: ChatState) -> Dict[str, List[BaseMessage]]:
        """
        LLM node that processes messages and generates responses.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dict containing the new messages to add to state
            
        Raises:
            ChatbotError: If LLM invocation fails
        """
        try:
            logger.debug(f"Processing {len(state['messages'])} messages")
            
            # Get the latest user message for context
            if not state["messages"]:
                raise ChatbotError("No messages found in state")
            
            # Invoke the LLM with the conversation history
            response = self.llm.invoke(state["messages"])
            
            logger.debug(f"Generated response: {response.content[:100]}...")
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"LLM node processing failed: {e}")
            error_message = AIMessage(
                content=f"I apologize, but I encountered an error: {str(e)}. Please try again."
            )
            return {"messages": [error_message]}
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            str: Unique session identifier
        """
        import uuid
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id} for user: {user_id}")
        return session_id
    
    def chat(self, message: str, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Process a chat message and return the bot's response.
        
        Args:
            message: User's input message
            session_id: Session identifier
            user_id: Optional user identifier
            
        Returns:
            str: Bot's response message
            
        Raises:
            ChatbotError: If chat processing fails
        """
        try:
            if not message.strip():
                raise ChatbotError("Empty message provided")
            
            logger.info(f"Processing message from session {session_id}: {message[:50]}...")
            
            # Create initial state
            initial_state: ChatState = {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.now().isoformat()
            }
            
            # Process through the graph
            result = self.graph.invoke(initial_state)
            
            # Extract the bot's response
            if not result["messages"]:
                raise ChatbotError("No response generated")
            
            # Get the last message (should be the AI response)
            last_message = result["messages"][-1]
            if not isinstance(last_message, AIMessage):
                raise ChatbotError("Expected AI message in response")
            
            response = last_message.content
            if isinstance(response, str):
                response_text = response
            else:
                # Handle list or other content types
                response_text = str(response)
            
            logger.info(f"Generated response for session {session_id}: {response_text[:50]}...")
            
            return response_text
            
        except ChatbotError:
            raise
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            raise ChatbotError(f"Chat processing failed: {e}") from e
    
    def chat_with_history(
        self, 
        message: str, 
        conversation_history: List[BaseMessage], 
        session_id: str,
        user_id: Optional[str] = None
    ) -> tuple[str, List[BaseMessage]]:
        """
        Process a chat message with conversation history and return response with updated history.
        
        Args:
            message: User's input message
            conversation_history: Previous conversation messages
            session_id: Session identifier
            user_id: Optional user identifier
            
        Returns:
            tuple: (Bot's response, Updated conversation history)
            
        Raises:
            ChatbotError: If chat processing fails
        """
        try:
            if not message.strip():
                raise ChatbotError("Empty message provided")
            
            logger.info(f"Processing message with history from session {session_id}")
            
            # Add the new user message to history
            updated_history = conversation_history + [HumanMessage(content=message)]
            
            # Create state with full conversation history
            initial_state: ChatState = {
                "messages": updated_history,
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.now().isoformat()
            }
            
            # Process through the graph
            result = self.graph.invoke(initial_state)
            
            # Extract the bot's response
            if not result["messages"]:
                raise ChatbotError("No response generated")
            
            # Get the AI response (last message should be the new AI response)
            ai_response = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    ai_response = msg
                    break
            
            if not ai_response:
                raise ChatbotError("No AI response found in result")
            
            response_content = ai_response.content
            if isinstance(response_content, str):
                response_text = response_content
            else:
                # Handle list or other content types
                response_text = str(response_content)
            
            final_history = result["messages"]
            
            logger.info(f"Generated response with history for session {session_id}")
            
            return response_text, final_history
            
        except ChatbotError:
            raise
        except Exception as e:
            logger.error(f"Chat with history processing failed: {e}")
            raise ChatbotError(f"Chat with history processing failed: {e}") from e


def main() -> None:
    """
    Main function for interactive chatbot session.
    
    Provides a simple command-line interface for testing the chatbot.
    """
    try:
        print("🤖 Basic Chatbot - LangGraph Implementation")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'help' for available commands")
        print()
        
        # Initialize chatbot
        chatbot = BasicChatbot()
        session_id = chatbot.create_session()
        conversation_history: List[BaseMessage] = []
        
        print("Chatbot initialized successfully! Start chatting...")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'help':
                    print("Available commands:")
                    print("  - 'quit', 'exit', 'bye': End the conversation")
                    print("  - 'help': Show this help message")
                    print("  - 'clear': Clear conversation history")
                    print("  - Any other text: Chat with the bot")
                    continue
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("🧹 Conversation history cleared!")
                    continue
                elif not user_input:
                    print("Please enter a message or type 'help' for commands.")
                    continue
                
                # Get bot response with history
                response, conversation_history = chatbot.chat_with_history(
                    user_input, conversation_history, session_id
                )
                
                print(f"Bot: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            except ChatbotError as e:
                print(f"❌ Chatbot Error: {e}")
                print("Please try again or type 'help' for commands.")
            except Exception as e:
                print(f"❌ Unexpected Error: {e}")
                print("Please try again or type 'help' for commands.")
                
    except ChatbotError as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("Please check your configuration and try again.")
    except Exception as e:
        print(f"❌ Unexpected initialization error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()





