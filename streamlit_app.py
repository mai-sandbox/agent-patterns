"""
Streamlit App for LangGraph Agent Interaction and Feedback

This app provides a web interface for interacting with a LangGraph agent,
collecting user feedback, and dynamically updating the agent's configuration
based on that feedback.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
import streamlit as st
from dotenv import load_dotenv
from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LangGraph Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "agent_response_complete" not in st.session_state:
    st.session_state.agent_response_complete = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "client" not in st.session_state:
    st.session_state.client = None


def get_langgraph_client():
    """Initialize and return LangGraph SDK client."""
    try:
        client = get_client(url="http://localhost:2024")
        return client
    except Exception as e:
        st.error(f"Failed to connect to LangGraph server: {e}")
        st.error("Make sure to run 'langgraph dev' in your terminal first.")
        return None


async def create_or_get_assistant(client, system_prompt: str) -> Optional[str]:
    """Create a new assistant or get existing one with the given system prompt."""
    try:
        # Create a new assistant with the current system prompt
        assistant = await client.assistants.create(
            graph_id="agent",
            config={
                "configurable": {
                    "system_prompt": system_prompt
                }
            },
            name=f"Chat Assistant - {len(st.session_state.messages)}"
        )
        return assistant["assistant_id"]
    except Exception as e:
        st.error(f"Failed to create assistant: {e}")
        return None


async def update_assistant_config(client, assistant_id: str, new_system_prompt: str) -> bool:
    """Update the assistant configuration with new system prompt."""
    try:
        await client.assistants.update(
            assistant_id,
            config={
                "configurable": {
                    "system_prompt": new_system_prompt
                }
            }
        )
        return True
    except Exception as e:
        st.error(f"Failed to update assistant: {e}")
        return False


async def stream_agent_response(client, assistant_id: str, user_input: str) -> List[str]:
    """Stream agent response and return all chunks."""
    response_chunks = []
    
    try:
        # Create input for the agent
        input_data = {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }
        
        # Stream the response
        async for chunk in client.runs.stream(
            thread_id=None,  # Threadless run
            assistant_id=assistant_id,
            input=input_data,
            stream_mode="messages"
        ):
            if chunk.event == "messages/partial":
                # Handle partial message updates
                if chunk.data and len(chunk.data) > 0:
                    message = chunk.data[-1]
                    if hasattr(message, 'content') and message.content:
                        response_chunks.append(message.content)
            elif chunk.event == "messages/complete":
                # Handle complete message
                if chunk.data and len(chunk.data) > 0:
                    message = chunk.data[-1]
                    if hasattr(message, 'content') and message.content:
                        response_chunks.append(message.content)
                        
    except Exception as e:
        st.error(f"Error streaming response: {e}")
        response_chunks.append(f"Error: {str(e)}")
    
    return response_chunks


def display_chat_messages():
    """Display chat messages in the main area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_feedback_and_update_prompt(feedback: str, current_prompt: str) -> str:
    """Process user feedback and generate an updated system prompt."""
    # Simple feedback processing - in a real app, you might use an LLM to process this
    feedback_lower = feedback.lower()
    
    # Basic feedback processing logic
    if "more helpful" in feedback_lower or "be more helpful" in feedback_lower:
        if "helpful" not in current_prompt.lower():
            return current_prompt + " Be extra helpful and provide detailed explanations."
    
    elif "shorter" in feedback_lower or "more concise" in feedback_lower:
        if "concise" not in current_prompt.lower():
            return current_prompt + " Keep responses concise and to the point."
    
    elif "more detailed" in feedback_lower or "more information" in feedback_lower:
        if "detailed" not in current_prompt.lower():
            return current_prompt + " Provide comprehensive and detailed responses."
    
    elif "friendlier" in feedback_lower or "more friendly" in feedback_lower:
        if "friendly" not in current_prompt.lower():
            return current_prompt + " Be warm, friendly, and conversational."
    
    elif "professional" in feedback_lower or "more formal" in feedback_lower:
        if "professional" not in current_prompt.lower():
            return current_prompt + " Maintain a professional and formal tone."
    
    else:
        # Generic feedback incorporation
        return current_prompt + f" User feedback: {feedback}"
    
    return current_prompt


def main():
    """Main Streamlit application."""
    st.title("🤖 LangGraph Agent Chat")
    st.markdown("Chat with an AI agent powered by LangGraph with real-time feedback and configuration updates.")
    
    # Initialize client
    if st.session_state.client is None:
        st.session_state.client = get_langgraph_client()
    
    if st.session_state.client is None:
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System prompt configuration
        default_prompt = "You are a helpful AI assistant. Answer questions clearly and concisely."
        system_prompt = st.text_area(
            "System Prompt",
            value=default_prompt,
            height=150,
            help="Configure how the agent should behave and respond."
        )
        
        # Connection status
        st.header("🔗 Connection Status")
        if st.session_state.client:
            st.success("✅ Connected to LangGraph server")
        else:
            st.error("❌ Not connected to LangGraph server")
        
        # Instructions
        st.header("📋 Instructions")
        st.markdown("""
        1. **Configure** the system prompt above
        2. **Chat** with the agent in the main area
        3. **Provide feedback** after each response
        4. **Update & Rerun** to improve the agent
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.assistant_id = None
            st.session_state.last_user_input = ""
            st.session_state.agent_response_complete = False
            st.session_state.feedback_submitted = False
            st.rerun()
    
    # Main chat interface
    display_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_user_input = prompt
        st.session_state.agent_response_complete = False
        st.session_state.feedback_submitted = False
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Create or get assistant
        if st.session_state.assistant_id is None:
            with st.spinner("Creating assistant..."):
                assistant_id = asyncio.run(
                    create_or_get_assistant(st.session_state.client, system_prompt)
                )
                st.session_state.assistant_id = assistant_id
        
        # Stream agent response
        if st.session_state.assistant_id:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                with st.spinner("Agent is thinking..."):
                    response_chunks = asyncio.run(
                        stream_agent_response(
                            st.session_state.client,
                            st.session_state.assistant_id,
                            prompt
                        )
                    )
                
                # Display the response (simulate streaming effect)
                for chunk in response_chunks:
                    if chunk and chunk != full_response:
                        full_response = chunk
                        message_placeholder.markdown(full_response + "▌")
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.agent_response_complete = True
    
    # Feedback section (appears after agent response)
    if st.session_state.agent_response_complete and not st.session_state.feedback_submitted:
        st.markdown("---")
        st.subheader("💬 Provide Feedback")
        
        with st.form("feedback_form"):
            feedback = st.text_area(
                "How can the agent improve its response?",
                placeholder="e.g., 'Be more helpful', 'Provide shorter answers', 'Be more detailed', etc.",
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submit_feedback = st.form_submit_button("📝 Submit Feedback", type="primary")
            with col2:
                skip_feedback = st.form_submit_button("⏭️ Skip Feedback", type="secondary")
            
            if submit_feedback and feedback.strip():
                # Process feedback and update system prompt
                new_prompt = process_feedback_and_update_prompt(feedback, system_prompt)
                
                # Update assistant configuration
                with st.spinner("Updating agent configuration..."):
                    success = asyncio.run(
                        update_assistant_config(
                            st.session_state.client,
                            st.session_state.assistant_id,
                            new_prompt
                        )
                    )
                
                if success:
                    st.success("✅ Agent configuration updated!")
                    st.session_state.feedback_submitted = True
                    
                    # Show updated prompt
                    with st.expander("🔄 Updated System Prompt"):
                        st.text_area("New prompt:", value=new_prompt, height=100, disabled=True)
                    
                    # Option to rerun with updated configuration
                    if st.button("🔄 Rerun Previous Input", type="primary"):
                        if st.session_state.last_user_input:
                            # Remove the last assistant response
                            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                                st.session_state.messages.pop()
                            
                            # Re-add user message and get new response
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                                full_response = ""
                                
                                with st.spinner("Agent is responding with updated configuration..."):
                                    response_chunks = asyncio.run(
                                        stream_agent_response(
                                            st.session_state.client,
                                            st.session_state.assistant_id,
                                            st.session_state.last_user_input
                                        )
                                    )
                                
                                # Display the new response
                                for chunk in response_chunks:
                                    if chunk and chunk != full_response:
                                        full_response = chunk
                                        message_placeholder.markdown(full_response + "▌")
                                
                                message_placeholder.markdown(full_response)
                                
                                # Add new response to chat history
                                if full_response:
                                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                                    st.success("✅ Response updated with new configuration!")
                else:
                    st.error("❌ Failed to update agent configuration.")
            
            elif skip_feedback:
                st.session_state.feedback_submitted = True
                st.info("Feedback skipped. You can continue chatting.")


if __name__ == "__main__":
    main()
