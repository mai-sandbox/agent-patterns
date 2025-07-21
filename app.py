import streamlit as st
import uuid
from langgraph_sdk import get_sync_client
from typing import Dict, Any, List
import time

# Configure Streamlit page
st.set_page_config(
    page_title="LangGraph Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI assistant with access to tools. Use them when appropriate to help users."
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "agent_response_complete" not in st.session_state:
    st.session_state.agent_response_complete = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Connect to LangGraph server
@st.cache_resource
def get_langgraph_client():
    """Get LangGraph client connection."""
    try:
        client = get_sync_client(url="http://localhost:2024")
        return client
    except Exception as e:
        st.error(f"Failed to connect to LangGraph server: {e}")
        st.error("Make sure the LangGraph server is running with 'langgraph dev'")
        return None

def stream_agent_response(client, user_input: str, system_prompt: str, thread_id: str):
    """Stream agent response and display in real-time."""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "system_prompt": system_prompt
        }
    }
    
    # Create a placeholder for streaming content
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        # Stream the agent response
        for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id="agent",
            input={"messages": [{"role": "human", "content": user_input}]},
            config=config,
            stream_mode="messages-tuple"
        ):
            if chunk.event == "messages/partial":
                # Extract message content from the streaming chunk
                if chunk.data and len(chunk.data) > 0:
                    message_data = chunk.data[-1]  # Get the latest message
                    if hasattr(message_data, 'content') and message_data.content:
                        full_response = message_data.content
                        response_placeholder.markdown(f"**🤖 Assistant:** {full_response}")
            elif chunk.event == "messages/complete":
                # Final message received
                if chunk.data and len(chunk.data) > 0:
                    message_data = chunk.data[-1]
                    if hasattr(message_data, 'content') and message_data.content:
                        full_response = message_data.content
                        response_placeholder.markdown(f"**🤖 Assistant:** {full_response}")
                        
        return full_response
        
    except Exception as e:
        st.error(f"Error streaming response: {e}")
        return None

def rerun_with_updated_config(client, user_input: str, updated_system_prompt: str, thread_id: str):
    """Rerun the previous input with updated configuration."""
    st.info("🔄 Rerunning with updated configuration...")
    
    # Create new thread for the rerun to avoid confusion
    new_thread_id = str(uuid.uuid4())
    
    response = stream_agent_response(client, user_input, updated_system_prompt, new_thread_id)
    
    if response:
        # Add to session state messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.agent_response_complete = True
        st.session_state.system_prompt = updated_system_prompt
        st.success("✅ Rerun completed with updated configuration!")

# Main UI
st.title("🤖 LangGraph Agent Chat")
st.markdown("Chat with a LangGraph agent and provide feedback to improve its responses!")

# Get client
client = get_langgraph_client()

if client is None:
    st.stop()

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # System prompt configuration
    st.subheader("System Prompt")
    current_system_prompt = st.text_area(
        "Current system prompt:",
        value=st.session_state.system_prompt,
        height=150,
        help="This is the current system prompt that guides the agent's behavior."
    )
    
    if st.button("Update System Prompt"):
        st.session_state.system_prompt = current_system_prompt
        st.success("System prompt updated!")
        st.rerun()
    
    st.divider()
    
    # Thread information
    st.subheader("Session Info")
    st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
    
    if st.button("New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.agent_response_complete = False
        st.session_state.feedback_submitted = False
        st.success("Started new conversation!")
        st.rerun()

# Main chat interface
st.header("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**👤 You:** {message['content']}")
    else:
        st.markdown(f"**🤖 Assistant:** {message['content']}")

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message
    st.markdown(f"**👤 You:** {user_input}")
    
    # Store the user input for potential rerun
    st.session_state.last_user_input = user_input
    
    # Stream agent response
    with st.spinner("🤔 Agent is thinking..."):
        response = stream_agent_response(
            client, 
            user_input, 
            st.session_state.system_prompt, 
            st.session_state.thread_id
        )
    
    if response:
        # Add messages to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.agent_response_complete = True
        st.session_state.feedback_submitted = False
        st.rerun()

# Feedback section (appears after agent response)
if st.session_state.agent_response_complete and st.session_state.last_user_input:
    st.divider()
    st.header("📝 Feedback")
    
    if not st.session_state.feedback_submitted:
        st.markdown("**How was the agent's response? Provide feedback to improve future interactions:**")
        
        feedback = st.text_area(
            "Your feedback:",
            placeholder="E.g., 'The response was too verbose, please be more concise' or 'Please be more friendly and use emojis'",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Submit Feedback", type="primary"):
                if feedback.strip():
                    # Process feedback and update system prompt
                    updated_prompt = f"{st.session_state.system_prompt}\n\nUser feedback: {feedback}\nPlease incorporate this feedback into your future responses."
                    
                    st.session_state.system_prompt = updated_prompt
                    st.session_state.feedback_submitted = True
                    
                    st.success("✅ Feedback submitted! System prompt has been updated.")
                    st.info("You can now rerun the previous input with the updated configuration.")
                    st.rerun()
                else:
                    st.warning("Please provide some feedback before submitting.")
        
        with col2:
            if st.button("⏭️ Skip Feedback"):
                st.session_state.feedback_submitted = True
                st.info("Feedback skipped.")
                st.rerun()
    
    else:
        # Show rerun option after feedback is submitted
        st.success("✅ Feedback has been incorporated into the system prompt!")
        
        st.markdown("**Updated System Prompt:**")
        st.text_area(
            "Current system prompt:",
            value=st.session_state.system_prompt,
            height=150,
            disabled=True
        )
        
        if st.button("🔄 Rerun with Updated Config", type="primary"):
            rerun_with_updated_config(
                client,
                st.session_state.last_user_input,
                st.session_state.system_prompt,
                st.session_state.thread_id
            )
            st.rerun()

# Footer
st.divider()
st.markdown(
    """<div style='text-align: center; color: gray;'>
    <small>🚀 Powered by LangGraph | Make sure to run <code>langgraph dev</code> to start the agent server</small>
    </div>""",
    unsafe_allow_html=True
)
