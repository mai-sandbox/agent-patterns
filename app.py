"""
Streamlit App for LangGraph Agent Interaction

This app connects to a LangGraph agent deployed using 'langgraph dev' and provides
a user interface for interacting with the agent, including real-time streaming
of responses and system prompt configuration.
"""

import streamlit as st
import asyncio
from langgraph_sdk import get_client
from typing import Dict, Any, List
import json
import time

# Page configuration
st.set_page_config(
    page_title="LangGraph Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI assistant with access to web search. Use tools when needed to provide accurate and up-to-date information."
if "client" not in st.session_state:
    try:
        st.session_state.client = get_client(url="http://localhost:2024")
    except Exception as e:
        st.session_state.client = None
        st.session_state.connection_error = str(e)
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


def stream_agent_response(client, user_input: str, system_prompt: str) -> List[Dict[str, Any]]:
    """
    Stream agent response using LangGraph SDK client.
    
    Args:
        client: LangGraph SDK client
        user_input: User's input message
        system_prompt: Current system prompt configuration
        
    Returns:
        List of response chunks from the agent
    """
    try:
        # Prepare the input for the agent
        input_data = {
            "messages": [
                {"role": "human", "content": user_input}
            ]
        }
        
        # Configuration with system prompt
        config = {
            "system_prompt": system_prompt
        }
        
        # Stream the response
        response_chunks = []
        for chunk in client.runs.stream(
            thread_id=None,  # Threadless run
            assistant_id="agent",
            input=input_data,
            config=config,
            stream_mode="messages-tuple"
        ):
            response_chunks.append(chunk)
        
        return response_chunks
    
    except Exception as e:
        st.error(f"Error streaming response: {str(e)}")
        return []


def display_message(role: str, content: str):
    """Display a message in the chat interface."""
    with st.chat_message(role):
        st.write(content)


def main():
    """Main Streamlit application."""
    st.title("🤖 LangGraph Agent Chat")
    st.markdown("Connect to your LangGraph agent and chat with real-time streaming!")
    
    # Check connection status
    if st.session_state.client is None:
        st.error("❌ Failed to connect to LangGraph server")
        st.error(f"Error: {st.session_state.get('connection_error', 'Unknown error')}")
        st.info("Make sure your LangGraph server is running with `langgraph dev`")
        return
    else:
        st.success("✅ Connected to LangGraph server at http://localhost:2024")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System prompt configuration
        st.subheader("System Prompt")
        new_system_prompt = st.text_area(
            "Configure the agent's system prompt:",
            value=st.session_state.system_prompt,
            height=150,
            help="This prompt defines how the agent behaves and responds to queries."
        )
        
        if st.button("Update System Prompt"):
            st.session_state.system_prompt = new_system_prompt
            st.success("System prompt updated!")
            st.rerun()
        
        # Display current system prompt
        st.info(f"**Current System Prompt:**
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])
    
    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message("user", user_input)
        
        # Stream agent response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Show loading indicator
            with st.spinner("Agent is thinking..."):
                response_chunks = stream_agent_response(
                    st.session_state.client, 
                    user_input, 
                    st.session_state.system_prompt
                )
            
            # Process and display streaming response
            for chunk in response_chunks:
                if chunk.event == "messages/partial":
                    # Extract message content from the chunk
                    if hasattr(chunk, 'data') and chunk.data:
                        if isinstance(chunk.data, tuple) and len(chunk.data) > 1:
                            message_data = chunk.data[1]  # Second element contains the message
                            if hasattr(message_data, 'content'):
                                content = message_data.content
                                if isinstance(content, str):
                                    full_response = content
                                elif isinstance(content, list):
                                    # Handle content blocks
                                    for block in content:
                                        if hasattr(block, 'text'):
                                            full_response += block.text
                                        elif isinstance(block, dict) and 'text' in block:
                                            full_response += block['text']
                                
                                # Update the display in real-time
                                message_placeholder.markdown(full_response)
                
                elif chunk.event == "messages/complete":
                    # Final message processing
                    if hasattr(chunk, 'data') and chunk.data:
                        if isinstance(chunk.data, tuple) and len(chunk.data) > 1:
                            message_data = chunk.data[1]
                            if hasattr(message_data, 'content'):
                                content = message_data.content
                                if isinstance(content, str):
                                    full_response = content
                                elif isinstance(content, list):
                                    # Handle content blocks
                                    text_content = ""
                                    for block in content:
                                        if hasattr(block, 'text'):
                                            text_content += block.text
                                        elif isinstance(block, dict) and 'text' in block:
                                            text_content += block['text']
                                    full_response = text_content
            
            # Ensure we have a response to display
            if not full_response:
                full_response = "I apologize, but I couldn't generate a proper response. Please try again."
            
            # Final display update
            message_placeholder.markdown(full_response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()


