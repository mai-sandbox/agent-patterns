"""
Streamlit application for LangGraph Agent Interaction and Feedback.

This app connects to a LangGraph agent deployed using 'langgraph dev' and provides:
- User interface for sending messages to the agent
- Real-time streaming of agent responses
- Feedback collection after agent runs
- Configuration updates based on feedback
- Rerun capability with updated configuration
"""

import asyncio
import os
from typing import Dict, Any, Optional, List
import json
import time

import streamlit as st
import nest_asyncio
from langgraph_sdk import get_client
from dotenv import load_dotenv

# Apply nest_asyncio to handle Streamlit's event loop
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_LANGGRAPH_URL = "http://localhost:2024"
ASSISTANT_ID = "agent"  # Must match the name in langgraph.json


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Static values
        "messages": [],
        "client": None,
        "thread_id": None,
        
        # Dynamic tracking - prefix with 'current_'
        "current_system_prompt": "You are a helpful AI assistant. Use the search tool when you need current information or facts that you're not certain about.",
        "current_config": {},
        "current_user_input": "",
        
        # UI state
        "show_feedback": False,
        "last_user_input": None,
        "agent_response": "",
        "feedback_submitted": False,
        "run_completed": False,
        
        # Streaming state
        "streaming_placeholder": None,
        "streaming_content": "",
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_langgraph_client():
    """Get or create LangGraph SDK client."""
    if st.session_state.client is None:
        langgraph_url = os.getenv("LANGGRAPH_API_URL", DEFAULT_LANGGRAPH_URL)
        try:
            st.session_state.client = get_client(url=langgraph_url)
        except Exception as e:
            st.error(f"Failed to connect to LangGraph server at {langgraph_url}: {e}")
            st.info("Make sure the LangGraph server is running with `langgraph dev`")
            return None
    return st.session_state.client


async def stream_agent_response(client, user_input: str, system_prompt: str) -> str:
    """
    Stream agent response and return the complete response.
    
    Args:
        client: LangGraph SDK client
        user_input: User's message
        system_prompt: Current system prompt configuration
        
    Returns:
        Complete agent response as string
    """
    try:
        # Create configuration with system prompt
        config = {
            "configurable": {
                "system_prompt": system_prompt
            }
        }
        
        # Prepare input messages
        input_data = {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }
        
        complete_response = ""
        
        # Stream the agent run
        async for chunk in client.runs.stream(
            thread_id=None,  # Stateless run
            assistant_id=ASSISTANT_ID,
            input=input_data,
            config=config,
            stream_mode="updates"
        ):
            if chunk.event == "updates" and chunk.data:
                # Handle different types of updates
                for node_name, node_data in chunk.data.items():
                    if "messages" in node_data:
                        messages = node_data["messages"]
                        if messages:
                            latest_message = messages[-1]
                            if hasattr(latest_message, 'content'):
                                content = latest_message.content
                            elif isinstance(latest_message, dict):
                                content = latest_message.get('content', '')
                            else:
                                content = str(latest_message)
                            
                            if content and content not in complete_response:
                                complete_response += content
                                # Update streaming placeholder
                                if st.session_state.streaming_placeholder:
                                    st.session_state.streaming_placeholder.markdown(complete_response)
        
        return complete_response
        
    except Exception as e:
        error_msg = f"Error streaming agent response: {e}"
        st.error(error_msg)
        return error_msg


def process_feedback_and_update_config(feedback: str, current_prompt: str) -> str:
    """
    Process user feedback and generate an updated system prompt.
    
    Args:
        feedback: User's feedback on the agent response
        current_prompt: Current system prompt
        
    Returns:
        Updated system prompt based on feedback
    """
    # Simple feedback processing - in a real application, you might use an LLM to process this
    feedback_lower = feedback.lower()
    
    # Basic feedback processing rules
    if "more detailed" in feedback_lower or "more information" in feedback_lower:
        if "provide detailed explanations" not in current_prompt:
            return current_prompt + " Always provide detailed explanations and comprehensive information."
    
    elif "shorter" in feedback_lower or "concise" in feedback_lower:
        if "be concise" not in current_prompt:
            return current_prompt + " Be concise and to the point in your responses."
    
    elif "more examples" in feedback_lower:
        if "include examples" not in current_prompt:
            return current_prompt + " Include relevant examples to illustrate your points."
    
    elif "more formal" in feedback_lower or "professional" in feedback_lower:
        if "professional tone" not in current_prompt:
            return current_prompt + " Use a professional and formal tone."
    
    elif "more casual" in feedback_lower or "friendly" in feedback_lower:
        if "friendly and casual" not in current_prompt:
            return current_prompt + " Use a friendly and casual tone."
    
    elif "search more" in feedback_lower or "use tools" in feedback_lower:
        if "always search" not in current_prompt:
            return current_prompt + " Always use the search tool to find the most current information before responding."
    
    else:
        # Generic feedback incorporation
        return current_prompt + f" User feedback: {feedback}"
    
    return current_prompt


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LangGraph Agent Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    st.title("🤖 LangGraph Agent Chat with Feedback")
    st.markdown("Connect to your LangGraph agent and provide feedback to improve responses!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System prompt configuration
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "Current System Prompt:",
            value=st.session_state.current_system_prompt,
            height=150,
            help="This prompt guides the agent's behavior and can be updated based on feedback."
        )
        
        if system_prompt != st.session_state.current_system_prompt:
            st.session_state.current_system_prompt = system_prompt
        
        # Connection status
        st.subheader("Connection Status")
        client = get_langgraph_client()
        if client:
            st.success("✅ Connected to LangGraph server")
        else:
            st.error("❌ Not connected to LangGraph server")
            st.info("Start the server with: `langgraph dev`")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # User input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your message:",
                height=100,
                placeholder="Ask me anything! I can search the web for current information."
            )
            submitted = st.form_submit_button("Send Message", type="primary")
        
        # Handle form submission
        if submitted and user_input.strip() and client:
            st.session_state.current_user_input = user_input
            st.session_state.last_user_input = user_input
            st.session_state.show_feedback = False
            st.session_state.feedback_submitted = False
            st.session_state.run_completed = False
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Display agent response with streaming
            with st.chat_message("assistant"):
                st.session_state.streaming_placeholder = st.empty()
                st.session_state.streaming_placeholder.markdown("🤔 Thinking...")
                
                # Stream the response
                try:
                    response = asyncio.run(stream_agent_response(
                        client, 
                        user_input, 
                        st.session_state.current_system_prompt
                    ))
                    st.session_state.agent_response = response
                    st.session_state.run_completed = True
                    st.session_state.show_feedback = True
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Show previous conversation if exists
        elif st.session_state.agent_response and st.session_state.last_user_input:
            with st.chat_message("user"):
                st.write(st.session_state.last_user_input)
            with st.chat_message("assistant"):
                st.write(st.session_state.agent_response)
    
    with col2:
        st.header("📝 Feedback & Actions")
        
        # Feedback section (only show after agent response)
        if st.session_state.show_feedback and st.session_state.run_completed:
            st.subheader("How was the response?")
            
            feedback = st.text_area(
                "Your feedback:",
                height=100,
                placeholder="Tell me how I can improve my response...",
                help="Your feedback will be used to update the system prompt."
            )
            
            col_feedback1, col_feedback2 = st.columns(2)
            
            with col_feedback1:
                if st.button("Submit Feedback", type="primary"):
                    if feedback.strip():
                        # Process feedback and update system prompt
                        updated_prompt = process_feedback_and_update_config(
                            feedback, 
                            st.session_state.current_system_prompt
                        )
                        st.session_state.current_system_prompt = updated_prompt
                        st.session_state.feedback_submitted = True
                        st.success("✅ Feedback submitted! System prompt updated.")
                        st.rerun()
                    else:
                        st.warning("Please enter some feedback first.")
            
            with col_feedback2:
                if st.button("Skip Feedback"):
                    st.session_state.feedback_submitted = True
                    st.rerun()
        
        # Rerun section (show after feedback is submitted)
        if st.session_state.feedback_submitted and st.session_state.last_user_input:
            st.subheader("🔄 Rerun with Updated Config")
            st.info("The system prompt has been updated based on your feedback.")
            
            if st.button("Rerun Previous Input", type="secondary"):
                if client:
                    # Reset states for rerun
                    st.session_state.show_feedback = False
                    st.session_state.feedback_submitted = False
                    st.session_state.run_completed = False
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(st.session_state.last_user_input)
                    
                    # Display agent response with streaming
                    with st.chat_message("assistant"):
                        st.session_state.streaming_placeholder = st.empty()
                        st.session_state.streaming_placeholder.markdown("🤔 Thinking with updated configuration...")
                        
                        # Stream the response with updated config
                        try:
                            response = asyncio.run(stream_agent_response(
                                client, 
                                st.session_state.last_user_input, 
                                st.session_state.current_system_prompt
                            ))
                            st.session_state.agent_response = response
                            st.session_state.run_completed = True
                            st.session_state.show_feedback = True
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        # Debug information (collapsible)
        with st.expander("🔍 Debug Information"):
            st.json({
                "current_system_prompt": st.session_state.current_system_prompt,
                "last_user_input": st.session_state.last_user_input,
                "show_feedback": st.session_state.show_feedback,
                "feedback_submitted": st.session_state.feedback_submitted,
                "run_completed": st.session_state.run_completed,
            })


if __name__ == "__main__":
    main()
