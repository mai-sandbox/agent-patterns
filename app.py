import streamlit as st
import json
import time
from typing import Dict, Any, Optional
from langgraph_sdk import get_sync_client
from langgraph_sdk.schema import Config
import requests

# Page configuration
st.set_page_config(
    page_title="LangGraph Agent Interaction",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "client" not in st.session_state:
    st.session_state.client = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_config" not in st.session_state:
    st.session_state.last_config = {}
if "run_completed" not in st.session_state:
    st.session_state.run_completed = False
if "agent_response" not in st.session_state:
    st.session_state.agent_response = []
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = "agent"

def initialize_client():
    """Initialize the LangGraph client connection"""
    try:
        client = get_sync_client(url="http://localhost:2024")
        # Test connection
        client.assistants.search()
        return client
    except Exception as e:
        st.error(f"Failed to connect to LangGraph server: {str(e)}")
        st.error("Make sure the LangGraph server is running with 'langgraph dev'")
        return None

def create_thread(client):
    """Create a new thread for conversation persistence"""
    try:
        thread = client.threads.create()
        return thread["thread_id"]
    except Exception as e:
        st.error(f"Failed to create thread: {str(e)}")
        return None

def stream_agent_run(client, thread_id, assistant_id, input_data, config):
    """Stream agent run and display updates"""
    try:
        # Create a placeholder for streaming updates
        stream_placeholder = st.empty()
        updates = []
        
        # Stream the agent run
        for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=input_data,
            config=config,
            stream_mode="updates"
        ):
            updates.append(chunk)
            # Display the current updates
            with stream_placeholder.container():
                st.subheader("🔄 Agent Progress")
                for i, update in enumerate(updates):
                    if hasattr(update, 'data') and update.data:
                        st.write(f"**Step {i+1}:** {update.data}")
                    else:
                        st.write(f"**Step {i+1}:** {str(update)}")
        
        return updates
    except Exception as e:
        st.error(f"Error during agent run: {str(e)}")
        return []

def update_agent_config(client, assistant_id, feedback, current_config):
    """Update agent configuration based on user feedback"""
    try:
        # Create an updated configuration that incorporates the feedback
        updated_config = current_config.copy()
        
        # Add feedback to the configuration
        if "configurable" not in updated_config:
            updated_config["configurable"] = {}
        
        # Store feedback in configuration
        updated_config["configurable"]["user_feedback"] = feedback
        updated_config["configurable"]["feedback_timestamp"] = time.time()
        
        # Update the assistant configuration
        client.assistants.update(
            assistant_id=assistant_id,
            config=updated_config
        )
        
        return updated_config
    except Exception as e:
        st.error(f"Failed to update agent configuration: {str(e)}")
        return current_config

# Main application
st.title("🤖 LangGraph Agent Interaction & Feedback")
st.markdown("Connect to your LangGraph agent, run it with custom configuration, and provide feedback to improve its performance.")

# Sidebar for connection and configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Connection section
    st.subheader("Connection")
    server_url = st.text_input("LangGraph Server URL", value="http://localhost:2024")
    
    if st.button("Connect to LangGraph Server"):
        with st.spinner("Connecting..."):
            st.session_state.client = initialize_client()
            if st.session_state.client:
                st.success("✅ Connected successfully!")
                # Create a new thread
                st.session_state.thread_id = create_thread(st.session_state.client)
                if st.session_state.thread_id:
                    st.success(f"📝 Thread created: {st.session_state.thread_id}")
    
    # Agent configuration
    st.subheader("Agent Settings")
    assistant_id = st.text_input("Assistant ID", value=st.session_state.assistant_id)
    st.session_state.assistant_id = assistant_id
    
    # Custom configuration
    st.subheader("Custom Configuration")
    config_json = st.text_area(
        "Configuration (JSON)",
        value=json.dumps(st.session_state.last_config, indent=2) if st.session_state.last_config else "{}",
        height=150
    )
    
    try:
        custom_config = json.loads(config_json)
        st.session_state.last_config = custom_config
    except json.JSONDecodeError:
        st.error("Invalid JSON configuration")
        custom_config = {}

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Agent Interaction")
    
    # Input section
    st.subheader("Input")
    user_input = st.text_area(
        "Enter your message for the agent:",
        value=st.session_state.last_input if st.session_state.last_input else "",
        height=100,
        placeholder="Type your message here..."
    )
    
    # Run agent button
    if st.button("🚀 Run Agent", type="primary", disabled=not st.session_state.client):
        if not user_input.strip():
            st.warning("Please enter a message for the agent.")
        else:
            st.session_state.last_input = user_input
            st.session_state.run_completed = False
            st.session_state.agent_response = []
            
            with st.spinner("Running agent..."):
                # Prepare input data
                input_data = {
                    "messages": [
                        {"role": "user", "content": user_input}
                    ]
                }
                
                # Stream the agent run
                updates = stream_agent_run(
                    st.session_state.client,
                    st.session_state.thread_id,
                    st.session_state.assistant_id,
                    input_data,
                    st.session_state.last_config
                )
                
                if updates:
                    st.session_state.agent_response = updates
                    st.session_state.run_completed = True
                    st.success("✅ Agent run completed!")

with col2:
    st.header("📊 Status")
    
    # Connection status
    if st.session_state.client:
        st.success("🟢 Connected")
    else:
        st.error("🔴 Not Connected")
    
    # Thread status
    if st.session_state.thread_id:
        st.info(f"📝 Thread: {st.session_state.thread_id[:8]}...")
    else:
        st.warning("📝 No Thread")
    
    # Run status
    if st.session_state.run_completed:
        st.success("✅ Run Completed")
    else:
        st.info("⏳ Ready to Run")

# Feedback section (only show after run completion)
if st.session_state.run_completed and st.session_state.agent_response:
    st.header("💭 Feedback & Configuration Update")
    
    with st.expander("📋 Agent Response Summary", expanded=True):
        for i, update in enumerate(st.session_state.agent_response):
            st.write(f"**Update {i+1}:** {str(update)}")
    
    # Feedback input
    feedback = st.text_area(
        "Provide feedback on the agent's performance:",
        placeholder="Enter your feedback here. This will be used to update the agent's configuration...",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Update Configuration", type="primary"):
            if feedback.strip():
                with st.spinner("Updating configuration..."):
                    updated_config = update_agent_config(
                        st.session_state.client,
                        st.session_state.assistant_id,
                        feedback,
                        st.session_state.last_config
                    )
                    st.session_state.last_config = updated_config
                    st.success("✅ Configuration updated with feedback!")
            else:
                st.warning("Please provide feedback before updating configuration.")
    
    with col2:
        if st.button("🔄 Rerun with Updated Config"):
            if st.session_state.last_input:
                st.rerun()
            else:
                st.warning("No previous input to rerun.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Make sure your LangGraph server is running with `langgraph dev` before connecting.")

