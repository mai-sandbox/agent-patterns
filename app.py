import streamlit as st
import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from langgraph_sdk import get_client, get_sync_client
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LangGraph Agent Interaction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "agent_url" not in st.session_state:
    st.session_state.agent_url = "http://localhost:2024"
if "agent_name" not in st.session_state:
    st.session_state.agent_name = "agent"
if "configuration" not in st.session_state:
    st.session_state.configuration = {}
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "agent_response" not in st.session_state:
    st.session_state.agent_response = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "client" not in st.session_state:
    st.session_state.client = None
if "connection_status" not in st.session_state:
    st.session_state.connection_status = "disconnected"

def check_agent_connection(url: str) -> bool:
    """Check if the LangGraph agent is accessible."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        try:
            # Try alternative health check endpoint
            response = requests.get(f"{url}/docs", timeout=5)
            return response.status_code == 200
        except:
            return False

def initialize_client(url: str):
    """Initialize the LangGraph client."""
    try:
        client = get_sync_client(url=url)
        st.session_state.client = client
        st.session_state.connection_status = "connected"
        return client
    except Exception as e:
        st.session_state.connection_status = "error"
        st.error(f"Failed to initialize client: {str(e)}")
        return None

def stream_agent_response(client, agent_name: str, input_data: Dict[str, Any], config: Dict[str, Any]):
    """Stream response from the LangGraph agent."""
    try:
        # Create a thread for persistence if needed
        thread = client.threads.create()
        thread_config = {
            "configurable": {
                "thread_id": thread["thread_id"],
                **config.get("configurable", {})
            }
        }
        
        # Stream the agent response
        response_container = st.empty()
        full_response = ""
        
        for stream_mode, chunk in client.runs.stream(
            thread["thread_id"],
            agent_name,
            input=input_data,
            config=thread_config,
            stream_mode=["values", "updates", "messages"]
        ):
            if stream_mode == 'updates' and chunk and hasattr(chunk, 'data'):
                chunk_data = str(chunk.data)
                full_response += chunk_data + "\n"
                response_container.markdown(f"```\n{full_response}\n```")
            elif chunk:
                chunk_str = str(chunk)
                full_response += chunk_str + "\n"
                response_container.markdown(f"```\n{full_response}\n```")
        
        return full_response, thread["thread_id"]
    except Exception as e:
        st.error(f"Error streaming agent response: {str(e)}")
        return None, None

def update_configuration_from_feedback(current_config: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    """Update configuration based on user feedback."""
    updated_config = current_config.copy()
    
    # Enhanced feedback processing with more sophisticated parsing
    feedback_lower = feedback.lower()
    
    if "configurable" not in updated_config:
        updated_config["configurable"] = {}
    
    # Add feedback as context
    updated_config["configurable"]["user_feedback"] = feedback
    updated_config["configurable"]["feedback_timestamp"] = datetime.now().isoformat()
    
    # Enhanced parameter extraction with better regex patterns
    import re
    
    # Temperature extraction with more flexible patterns
    if "temperature" in feedback_lower:
        try:
            # Multiple patterns for temperature extraction
            patterns = [
                r"temperature[\s:=]*(\d*\.?\d+)",
                r"temp[\s:=]*(\d*\.?\d+)",
                r"set.*temperature.*to[\s]*(\d*\.?\d+)",
                r"use.*temperature.*of[\s]*(\d*\.?\d+)"
            ]
            temp_match = None
            for pattern in patterns:
                temp_match = re.search(pattern, feedback_lower)
                if temp_match:
                    break
            if temp_match:
                temp_value = float(temp_match.group(1))
                # Validate temperature range (0.0 to 2.0 is typical for LLMs)
                if 0.0 <= temp_value <= 2.0:
                    updated_config["configurable"]["temperature"] = temp_value
                else:
                    st.warning(f"Temperature {temp_value} is outside typical range (0.0-2.0). Using default.")
        except:
            pass
    
    # Enhanced verbosity detection
    if "verbose" in feedback_lower or "detailed" in feedback_lower:
        updated_config["configurable"]["verbose"] = True
        updated_config["configurable"]["response_style"] = "detailed"
    
    if "concise" in feedback_lower or "brief" in feedback_lower:
        updated_config["configurable"]["verbose"] = False
        updated_config["configurable"]["response_style"] = "concise"
    
    # Max tokens extraction
    max_tokens_patterns = [
        r"max[\s_]*tokens[\s:=]*(\d+)",
        r"limit.*tokens.*to[\s]*(\d+)",
        r"use.*(\d+).*tokens"
    ]
    for pattern in max_tokens_patterns:
        match = re.search(pattern, feedback_lower)
        if match:
            try:
                tokens = int(match.group(1))
                if 1 <= tokens <= 32000:  # Reasonable token limits
                    updated_config["configurable"]["max_tokens"] = tokens
                break
            except:
                pass
    
    # Response style and tone detection
    if any(word in feedback_lower for word in ["creative", "imaginative", "original"]):
        updated_config["configurable"]["creativity"] = "high"
        if "temperature" not in updated_config["configurable"]:
            updated_config["configurable"]["temperature"] = 0.8
    
    if any(word in feedback_lower for word in ["factual", "accurate", "precise", "conservative"]):
        updated_config["configurable"]["creativity"] = "low"
        if "temperature" not in updated_config["configurable"]:
            updated_config["configurable"]["temperature"] = 0.2
    
    # Speed vs quality preferences
    if any(word in feedback_lower for word in ["faster", "quicker", "speed"]):
        updated_config["configurable"]["priority"] = "speed"
    
    if any(word in feedback_lower for word in ["quality", "better", "thorough"]):
        updated_config["configurable"]["priority"] = "quality"
    
    # Add feedback analysis summary
    updated_config["configurable"]["feedback_analysis"] = {
        "original_feedback": feedback,
        "extracted_preferences": {
            k: v for k, v in updated_config["configurable"].items() 
            if k not in ["user_feedback", "feedback_timestamp", "feedback_analysis"]
        }
    }
    
    return updated_config

# Main UI
st.title("🤖 LangGraph Agent Interaction & Feedback")
st.markdown("Connect to a LangGraph agent, run it with custom configuration, and provide feedback to improve future runs.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Agent connection settings
    st.subheader("Agent Connection")
    agent_url = st.text_input("Agent URL", value=st.session_state.agent_url)
    agent_name = st.text_input("Agent Name", value=st.session_state.agent_name)
    
    if st.button("Test Connection"):
        with st.spinner("Testing connection..."):
            if check_agent_connection(agent_url):
                st.success("✅ Connection successful!")
                st.session_state.agent_url = agent_url
                st.session_state.agent_name = agent_name
                initialize_client(agent_url)
            else:
                st.error("❌ Connection failed. Make sure the agent is running with `langgraph dev`.")
    
    # Configuration settings
    st.subheader("Agent Configuration")
    
    # JSON configuration editor
    config_text = st.text_area(
        "Configuration (JSON)",
        value=json.dumps(st.session_state.configuration, indent=2) if st.session_state.configuration else "{}",
        height=200,
        help="Enter configuration as JSON. This will be passed to the agent."
    )
    
    try:
        config = json.loads(config_text) if config_text.strip() else {}
        st.session_state.configuration = config
        st.success("✅ Valid JSON configuration")
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON: {str(e)}")
        config = st.session_state.configuration
    
    # Connection status
    st.subheader("Status")
    status_color = {
        "connected": "🟢",
        "disconnected": "🔴",
        "error": "🟠"
    }
    st.write(f"{status_color.get(st.session_state.connection_status, '🔴')} {st.session_state.connection_status.title()}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Agent Interaction")
    
    # Input for the agent
    user_input = st.text_area(
        "Enter your message for the agent:",
        height=100,
        placeholder="Type your message here..."
    )
    
    # Run agent button
    if st.button("🚀 Run Agent", type="primary", disabled=not user_input.strip()):
        if st.session_state.connection_status != "connected":
            st.error("Please establish a connection to the agent first.")
        else:
            st.session_state.last_input = user_input
            
            with st.spinner("Running agent..."):
                try:
                    # Prepare input data
                    input_data = {
                        "messages": [
                            {"role": "user", "content": user_input}
                        ]
                    }
                    
                    # Stream the response
                    response, thread_id = stream_agent_response(
                        st.session_state.client,
                        st.session_state.agent_name,
                        input_data,
                        st.session_state.configuration
                    )
                    
                    if response:
                        st.session_state.agent_response = response
                        st.session_state.run_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "input": user_input,
                            "response": response,
                            "config": st.session_state.configuration.copy(),
                            "thread_id": thread_id
                        })
                        st.success("✅ Agent run completed!")
                    
                except Exception as e:
                    st.error(f"Error running agent: {str(e)}")
    
    # Display agent response
    if st.session_state.agent_response:
        st.subheader("🤖 Agent Response")
        st.markdown(f"```\n{st.session_state.agent_response}\n```")

with col2:
    st.header("📝 Feedback & Rerun")
    
    if st.session_state.agent_response:
        # Feedback section
        st.subheader("💭 Provide Feedback")
        feedback = st.text_area(
            "How can the agent improve?",
            height=150,
            placeholder="Enter your feedback here. For example:\n- Be more concise\n- Provide more details\n- Use temperature 0.7\n- Be more creative",
            help="Your feedback will be used to update the agent configuration for future runs."
        )
        
        if st.button("📤 Submit Feedback", disabled=not feedback.strip()):
            st.session_state.feedback = feedback
            
            # Update configuration based on feedback
            updated_config = update_configuration_from_feedback(
                st.session_state.configuration,
                feedback
            )
            st.session_state.configuration = updated_config
            
            st.success("✅ Feedback submitted and configuration updated!")
            st.json(updated_config)
        
        # Rerun section
        if st.session_state.feedback and st.session_state.last_input:
            st.subheader("🔄 Rerun with Updated Config")
            st.write(f"**Previous input:** {st.session_state.last_input[:100]}{'...' if len(st.session_state.last_input) > 100 else ''}")
            
            if st.button("🔄 Rerun Agent", type="secondary"):
                if st.session_state.connection_status != "connected":
                    st.error("Please establish a connection to the agent first.")
                else:
                    with st.spinner("Rerunning agent with updated configuration..."):
                        try:
                            # Prepare input data
                            input_data = {
                                "messages": [
                                    {"role": "user", "content": st.session_state.last_input}
                                ]
                            }
                            
                            # Stream the response with updated config
                            response, thread_id = stream_agent_response(
                                st.session_state.client,
                                st.session_state.agent_name,
                                input_data,
                                st.session_state.configuration
                            )
                            
                            if response:
                                st.session_state.agent_response = response
                                st.session_state.run_history.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "input": st.session_state.last_input,
                                    "response": response,
                                    "config": st.session_state.configuration.copy(),
                                    "thread_id": thread_id,
                                    "rerun": True,
                                    "feedback": st.session_state.feedback
                                })
                                st.success("✅ Agent rerun completed with updated configuration!")
                                st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error rerunning agent: {str(e)}")

# Run history
if st.session_state.run_history:
    st.header("📊 Run History")
    
    with st.expander(f"View {len(st.session_state.run_history)} previous runs"):
        for i, run in enumerate(reversed(st.session_state.run_history)):
            st.subheader(f"Run {len(st.session_state.run_history) - i}")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.write(f"**Time:** {run['timestamp'][:19]}")
                if run.get('rerun'):
                    st.write("🔄 **Rerun**")
            
            with col2:
                st.write(f"**Input:** {run['input'][:50]}{'...' if len(run['input']) > 50 else ''}")
            
            with col3:
                if run.get('feedback'):
                    st.write(f"**Feedback:** {run['feedback'][:50]}{'...' if len(run['feedback']) > 50 else ''}")
            
            with st.expander("View full details"):
                st.write("**Full Input:**")
                st.text(run['input'])
                st.write("**Full Response:**")
                st.text(run['response'])
                st.write("**Configuration:**")
                st.json(run['config'])
                if run.get('feedback'):
                    st.write("**Feedback:**")
                    st.text(run['feedback'])
            
            st.divider()

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tip:** Make sure your LangGraph agent is running with `langgraph dev` before connecting. "
    "The default URL is http://localhost:2024"
)

