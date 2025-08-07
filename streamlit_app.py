"""
Streamlit application for LangGraph agent interaction and feedback.
Connects to a LangGraph agent deployed using 'langgraph dev'.
"""

import json
import os
from typing import Dict, Any
import streamlit as st
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LANGGRAPH_SERVER_URL = os.getenv("LANGGRAPH_SERVER_URL", "http://localhost:8123")
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant with access to tools."

# Page configuration
st.set_page_config(
    page_title="LangGraph Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LangGraphClient:
    """Client for interacting with LangGraph deployed agent."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)
    
    def invoke_agent(self, message: str, system_prompt: str) -> Dict[str, Any]:
        """
        Invoke the agent with a message and system prompt.
        
        Args:
            message: User message to send to the agent
            system_prompt: System prompt configuration
            
        Returns:
            Response from the agent
        """
        try:
            # Prepare the request payload
            payload = {
                "input": {
                    "messages": [{"type": "human", "content": message}],
                    "system_prompt": system_prompt
                },
                "config": {
                    "configurable": {
                        "system_prompt": system_prompt
                    }
                }
            }
            
            # Make request to LangGraph server
            response = self.client.post(
                f"{self.server_url}/agent/invoke",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Server returned status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def stream_agent(self, message: str, system_prompt: str):
        """
        Stream responses from the agent.
        
        Args:
            message: User message to send to the agent
            system_prompt: System prompt configuration
            
        Yields:
            Streaming responses from the agent
        """
        try:
            # Prepare the request payload
            payload = {
                "input": {
                    "messages": [{"type": "human", "content": message}],
                    "system_prompt": system_prompt
                },
                "config": {
                    "configurable": {
                        "system_prompt": system_prompt
                    }
                }
            }
            
            # Make streaming request to LangGraph server
            with self.client.stream(
                "POST",
                f"{self.server_url}/agent/stream",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    yield {"error": f"Server returned status {response.status_code}: {response.text}"}
                    return
                
                for line in response.iter_lines():
                    if line:
                        try:
                            # Parse streaming response
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield {"error": f"Connection error: {str(e)}"}
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    
    if "langgraph_client" not in st.session_state:
        st.session_state.langgraph_client = LangGraphClient(LANGGRAPH_SERVER_URL)
    
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
    
    if "agent_response" not in st.session_state:
        st.session_state.agent_response = ""
    
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True
    
    if "feedback_text" not in st.session_state:
        st.session_state.feedback_text = ""
    
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False


def display_chat_messages():
    """Display chat messages in the main area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def update_system_prompt_with_feedback(feedback: str, current_prompt: str) -> str:
    """
    Update the system prompt based on user feedback.
    
    Args:
        feedback: User feedback about the agent's response
        current_prompt: Current system prompt
        
    Returns:
        Updated system prompt incorporating the feedback
    """
    # Simple feedback integration - append feedback as additional instruction
    feedback_instruction = f"\n\nAdditional user guidance: {feedback.strip()}"
    
    # Avoid duplicate feedback by checking if similar feedback already exists
    if feedback.strip().lower() not in current_prompt.lower():
        updated_prompt = current_prompt + feedback_instruction
    else:
        updated_prompt = current_prompt
    
    return updated_prompt


def feedback_section():
    """Display the feedback collection interface."""
    st.subheader("💬 Provide Feedback")
    st.markdown("Help improve the agent's responses by providing feedback on the last interaction.")
    
    # Feedback text area
    feedback_text = st.text_area(
        "What would you like the agent to do differently next time?",
        value=st.session_state.feedback_text,
        height=100,
        placeholder="e.g., 'Be more concise', 'Provide more examples', 'Focus on practical solutions'...",
        help="Your feedback will be used to update the system prompt and improve future responses."
    )
    
    # Update session state with current feedback text
    st.session_state.feedback_text = feedback_text
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Submit feedback button
        if st.button("📝 Submit Feedback", type="primary", disabled=not feedback_text.strip()):
            if feedback_text.strip():
                # Update system prompt with feedback
                old_prompt = st.session_state.system_prompt
                updated_prompt = update_system_prompt_with_feedback(
                    feedback_text, 
                    st.session_state.system_prompt
                )
                
                st.session_state.system_prompt = updated_prompt
                st.session_state.feedback_submitted = True
                
                # Show success message
                st.success("✅ Feedback submitted! System prompt has been updated.")
                
                # Clear feedback text
                st.session_state.feedback_text = ""
                
                # Show what changed
                if updated_prompt != old_prompt:
                    with st.expander("📋 View System Prompt Changes"):
                        st.markdown("**Updated System Prompt:**")
                        st.text_area("", value=updated_prompt, height=150, disabled=True)
                
                st.rerun()
    
    with col2:
        # Rerun with updated config button
        if st.button(
            "🔄 Rerun with Updated Config", 
            type="secondary",
            disabled=not (st.session_state.feedback_submitted and st.session_state.last_user_input),
            help="Re-execute the previous input with the updated system prompt"
        ):
            if st.session_state.last_user_input:
                # Clear feedback state
                st.session_state.show_feedback = False
                st.session_state.feedback_submitted = False
                
                # Add a separator message to show the rerun
                st.session_state.messages.append({
                    "role": "system",
                    "content": "🔄 **Rerunning with updated configuration...**"
                })
                
                # Re-add the user input to trigger a new response
                st.session_state.messages.append({
                    "role": "user",
                    "content": st.session_state.last_user_input
                })
                
                st.rerun()


def configuration_panel():
    """Display the configuration panel in the sidebar."""
    st.sidebar.header("🔧 Configuration")
    
    # System prompt configuration
    st.sidebar.subheader("System Prompt")
    new_system_prompt = st.sidebar.text_area(
        "Edit the system prompt:",
        value=st.session_state.system_prompt,
        height=150,
        help="This defines how the agent behaves and responds to users."
    )
    
    # Update system prompt if changed
    if new_system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_system_prompt
        st.sidebar.success("System prompt updated!")
    
    # Connection status
    st.sidebar.subheader("Connection")
    st.sidebar.info(f"Server: {LANGGRAPH_SERVER_URL}")
    
    # Streaming configuration
    st.sidebar.subheader("Streaming")
    st.session_state.streaming_enabled = st.sidebar.checkbox(
        "Enable streaming responses",
        value=st.session_state.streaming_enabled,
        help="Stream responses in real-time for better user experience"
    )
    
    # Test connection button
    if st.sidebar.button("Test Connection"):
        try:
            response = httpx.get(f"{LANGGRAPH_SERVER_URL}/health", timeout=5.0)
            if response.status_code == 200:
                st.sidebar.success("✅ Connected to LangGraph server")
            else:
                st.sidebar.error(f"❌ Server responded with status {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")


def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🤖 LangGraph Agent Chat")
    st.markdown("Chat with a LangGraph agent that has tool calling capabilities!")
    
    # Configuration panel
    configuration_panel()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Conversation")
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            display_chat_messages()
        
        # User input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Store the user input
            st.session_state.last_user_input = user_input
            
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get agent response with streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                try:
                    if st.session_state.streaming_enabled:
                        # Use streaming for real-time response display
                        full_response = ""
                        agent_thinking = True
                        
                        # Show initial thinking indicator
                        progress_placeholder.info("🤔 Agent is thinking...")
                        
                        # Stream the response
                        for chunk in st.session_state.langgraph_client.stream_agent(
                            user_input, 
                            st.session_state.system_prompt
                        ):
                            if "error" in chunk:
                                error_msg = f"❌ Error: {chunk['error']}"
                                progress_placeholder.empty()
                                response_placeholder.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                "content": error_msg
                            })
                                break
                            
                            # Handle different streaming modes
                            if "event" in chunk:
                                event_type = chunk.get("event")
                                
                                # Handle agent progress updates
                                if event_type == "on_chain_start":
                                    if agent_thinking:
                                        progress_placeholder.info("🔧 Using tools...")
                                elif event_type == "on_tool_start":
                                    tool_name = chunk.get("name", "unknown tool")
                                    progress_placeholder.info(f"🛠️ Using {tool_name}...")
                                elif event_type == "on_tool_end":
                                    progress_placeholder.info("💭 Processing tool results...")
                                elif event_type == "on_chat_model_start":
                                    progress_placeholder.info("🧠 Generating response...")
                                    agent_thinking = False
                            
                            # Handle LLM token streaming
                            if "data" in chunk:
                                data = chunk["data"]
                                
                                # Handle different data types
                                if isinstance(data, dict):
                                    # Handle message chunks
                                    if "chunk" in data:
                                        chunk_data = data["chunk"]
                                        if isinstance(chunk_data, dict) and "content" in chunk_data:
                                            token = chunk_data["content"]
                                            if token:
                                                full_response += token
                                                # Update display with current response
                                                response_placeholder.markdown(full_response + "▌")
                                    
                                    # Handle complete messages
                                    elif "messages" in data:
                                        messages = data["messages"]
                                        for msg in messages:
                                            if msg.get("type") == "ai" or msg.get("role") == "assistant":
                                                content = msg.get("content", "")
                                                if content and content != full_response:
                                                    full_response = content
                                                    response_placeholder.markdown(full_response + "▌")
                            
                            # Handle custom updates
                            if "output" in chunk:
                                output = chunk["output"]
                                if "messages" in output:
                                    messages = output["messages"]
                                    # Find the last assistant message
                                    for msg in reversed(messages):
                                        if msg.get("type") == "ai" or msg.get("role") == "assistant":
                                            content = msg.get("content", "")
                                            if content:
                                                full_response = content
                                                break
                    
                    # Clear progress indicator and finalize response
                    progress_placeholder.empty()
                    
                    if full_response:
                        # Remove cursor and display final response
                        response_placeholder.markdown(full_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response
                        })
                        st.session_state.agent_response = full_response
                        st.session_state.show_feedback = True
                    else:
                        # Fallback to invoke method if streaming didn't work
                        progress_placeholder.info("🔄 Falling back to standard mode...")
                        response = st.session_state.langgraph_client.invoke_agent(
                            user_input, 
                            st.session_state.system_prompt
                        )
                        
                        progress_placeholder.empty()
                        
                        if "error" in response:
                            error_msg = f"❌ Error: {response['error']}"
                            response_placeholder.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                        else:
                            # Extract the assistant's response
                            if "output" in response and "messages" in response["output"]:
                                messages = response["output"]["messages"]
                                assistant_message = ""
                                
                                # Find the last assistant message
                                for msg in reversed(messages):
                                    if msg.get("type") == "ai" or msg.get("role") == "assistant":
                                        assistant_message = msg.get("content", "")
                                        break
                                
                                if assistant_message:
                                    response_placeholder.write(assistant_message)
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": assistant_message
                                    })
                                    st.session_state.agent_response = assistant_message
                                    st.session_state.show_feedback = True
                                else:
                                    error_msg = "❌ No response received from agent"
                                    response_placeholder.error(error_msg)
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": error_msg
                                    })
                            else:
                                error_msg = "❌ Unexpected response format from agent"
                                response_placeholder.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                
                except Exception as e:
                    progress_placeholder.empty()
                    error_msg = f"❌ Unexpected error: {str(e)}"
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        
        # Feedback collection system
        if st.session_state.show_feedback and st.session_state.agent_response:
            st.divider()
            feedback_section()
    
    with col2:
        st.subheader("📊 Session Info")
        
        # Display session statistics
        st.metric("Messages", len(st.session_state.messages))
        st.metric("System Prompt Length", len(st.session_state.system_prompt))
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.show_feedback = False
            st.rerun()
        
        # Display current system prompt (truncated)
        st.subheader("🎯 Current System Prompt")
        prompt_preview = st.session_state.system_prompt[:100]
        if len(st.session_state.system_prompt) > 100:
            prompt_preview += "..."
        st.text_area("", value=prompt_preview, height=100, disabled=True)


if __name__ == "__main__":
    main()




















