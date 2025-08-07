"""
Streamlit App for Interactive LangGraph Agent Deployment

This application provides a web interface for interacting with a LangGraph agent
deployed using langgraph dev. Features include:
- Connection to LangGraph server at http://localhost:2024
- Input form for user messages and system prompt configuration
- Streaming display of agent responses
- Feedback collection and configuration updates
- Rerun functionality with updated configuration
"""

import streamlit as st
import asyncio
import nest_asyncio
from typing import Dict, Any, Optional, List
from langgraph_sdk import get_client
import json
import time

# Apply nest_asyncio to handle Streamlit's event loop
nest_asyncio.apply()

# Configure Streamlit page
st.set_page_config(
    page_title="LangGraph Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Connection and client
        "client": None,
        "server_url": "http://localhost:2024",
        "assistant_id": "agent",
        # Chat state
        "messages": [],
        "current_user_input": "",
        "last_user_input": "",
        # Configuration
        "current_system_prompt": """You are a helpful AI assistant with access to tools. 
You can help users with weather information, mathematical calculations, and provide the current time.

When using tools:
- For weather: Ask for a specific location if not provided
- For calculations: Ensure the expression is mathematically valid
- Always provide clear, helpful responses

Be conversational and friendly while being accurate and helpful.""",
        # UI state
        "show_feedback": False,
        "feedback_text": "",
        "agent_response": "",
        "streaming_complete": False,
        "run_id": None,
        # Error handling
        "connection_error": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_langgraph_client():
    """Get or create LangGraph SDK client."""
    try:
        if st.session_state.client is None:
            st.session_state.client = get_client(url=st.session_state.server_url)
            st.session_state.connection_error = None
        return st.session_state.client
    except Exception as e:
        st.session_state.connection_error = str(e)
        return None


async def stream_agent_response(client, user_input: str, system_prompt: str) -> str:
    """Stream agent response and return the final response."""
    try:
        # Configuration for the agent run
        config = {"configurable": {"system_prompt": system_prompt}}

        # Create input for the agent
        input_data = {"messages": [{"role": "human", "content": user_input}]}

        # Create a placeholder for streaming content
        response_placeholder = st.empty()
        full_response = ""

        # Stream the agent run (following official LangGraph SDK pattern)
        async for chunk in client.runs.stream(
            None,  # Threadless run
            st.session_state.assistant_id,
            input=input_data,
            config=config,
        ):
            # Process streaming chunks following official documentation pattern
            if hasattr(chunk, "event") and hasattr(chunk, "data"):
                # Handle different event types as per LangGraph SDK documentation
                if chunk.event == "messages":
                    # Handle messages event - direct message streaming
                    if chunk.data and len(chunk.data) > 0:
                        message = chunk.data[-1]  # Get the latest message
                        if hasattr(message, "content"):
                            content = message.content
                        elif isinstance(message, dict) and "content" in message:
                            content = message["content"]
                        else:
                            content = str(message)

                        if content and content != full_response:
                            full_response = content
                            # Update the streaming display
                            response_placeholder.markdown(
                                f"**🤖 Assistant:** {full_response}"
                            )

                elif chunk.event == "updates":
                    # Handle updates event - node-based updates
                    if chunk.data:
                        for node_name, node_data in chunk.data.items():
                            if isinstance(node_data, dict) and "messages" in node_data:
                                messages = node_data["messages"]
                                if messages and len(messages) > 0:
                                    last_message = messages[-1]
                                    if hasattr(last_message, "content"):
                                        content = last_message.content
                                    elif (
                                        isinstance(last_message, dict)
                                        and "content" in last_message
                                    ):
                                        content = last_message["content"]
                                    else:
                                        continue

                                    if content and content != full_response:
                                        full_response = content
                                        # Update the streaming display
                                        response_placeholder.markdown(
                                            f"**🤖 Assistant:** {full_response}"
                                        )

                elif chunk.event == "metadata":
                    # Handle metadata events (run_id, etc.) - no display update needed
                    continue

                elif chunk.event == "error":
                    # Handle error events
                    error_msg = (
                        chunk.data.get("message", "Unknown error")
                        if chunk.data
                        else "Unknown error"
                    )
                    st.error(f"Agent error: {error_msg}")
                    return f"Error: {error_msg}"

        # If no response was streamed, return a default message
        if not full_response:
            full_response = "No response received from the agent."

        return full_response

    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            st.error(
                "⚠️ **API Key Required**: Please set your ANTHROPIC_API_KEY in the .env file to use the agent."
            )
            return "Error: API key not configured. Please check the .env file and add your ANTHROPIC_API_KEY."
        else:
            st.error(f"Error streaming agent response: {error_msg}")
            return f"Error: {error_msg}"


def update_system_prompt_from_feedback(feedback: str, current_prompt: str) -> str:
    """Update system prompt based on user feedback."""
    # Simple feedback-based prompt updates
    feedback_lower = feedback.lower()

    if "more detailed" in feedback_lower or "more information" in feedback_lower:
        updated_prompt = (
            current_prompt
            + "\n\nPlease provide detailed and comprehensive responses with additional context and explanations."
        )
    elif "shorter" in feedback_lower or "concise" in feedback_lower:
        updated_prompt = current_prompt + "\n\nKeep responses concise and to the point."
    elif "friendly" in feedback_lower or "casual" in feedback_lower:
        updated_prompt = (
            current_prompt + "\n\nUse a friendly, casual tone in your responses."
        )
    elif "formal" in feedback_lower or "professional" in feedback_lower:
        updated_prompt = (
            current_prompt
            + "\n\nMaintain a formal, professional tone in your responses."
        )
    elif "creative" in feedback_lower or "imaginative" in feedback_lower:
        updated_prompt = (
            current_prompt
            + "\n\nBe creative and imaginative in your responses when appropriate."
        )
    else:
        # Generic feedback incorporation
        updated_prompt = (
            current_prompt
            + f"\n\nUser feedback: {feedback}. Please adjust your responses accordingly."
        )

    return updated_prompt


def main():
    """Main Streamlit application."""
    init_session_state()

    # Header
    st.title("🤖 LangGraph Agent Chat")
    st.markdown("Interactive chat interface with configurable LangGraph agent")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Server connection status
        client = get_langgraph_client()
        if st.session_state.connection_error:
            st.error(f"❌ Connection Error: {st.session_state.connection_error}")
            st.info("Make sure the LangGraph server is running with: `langgraph dev`")
        else:
            st.success("✅ Connected to LangGraph Server")

        # System prompt configuration
        st.subheader("System Prompt")
        new_system_prompt = st.text_area(
            "Configure the agent's behavior:",
            value=st.session_state.current_system_prompt,
            height=200,
            help="This prompt defines how the agent behaves and responds to users.",
        )

        if st.button("Update System Prompt"):
            st.session_state.current_system_prompt = new_system_prompt
            st.success("System prompt updated!")

        # Server settings
        st.subheader("Server Settings")
        st.text_input("Server URL", value=st.session_state.server_url, disabled=True)
        st.text_input(
            "Assistant ID", value=st.session_state.assistant_id, disabled=True
        )

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 Chat")

        # Display chat history
        if st.session_state.messages:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"**👤 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['content']}")

        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your message:",
                placeholder="Ask me about the weather, calculations, or the current time...",
                height=100,
            )
            submitted = st.form_submit_button("Send Message", use_container_width=True)

        # Process user input
        if submitted and user_input.strip():
            if not client:
                st.error("Cannot send message: No connection to LangGraph server")
            else:
                # Store user input
                st.session_state.current_user_input = user_input
                st.session_state.last_user_input = user_input

                # Add user message to chat history
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )

                # Display user message
                st.markdown(f"**👤 You:** {user_input}")

                # Stream agent response
                with st.spinner("🤖 Agent is thinking..."):
                    try:
                        response = asyncio.run(
                            stream_agent_response(
                                client,
                                user_input,
                                st.session_state.current_system_prompt,
                            )
                        )

                        # Store agent response
                        st.session_state.agent_response = response
                        st.session_state.streaming_complete = True

                        # Add agent response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                        # Show feedback form
                        st.session_state.show_feedback = True

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col2:
        st.header("📝 Actions")

        # Feedback section
        if st.session_state.show_feedback and st.session_state.streaming_complete:
            st.subheader("Feedback")
            st.info(
                "How was the agent's response? Your feedback will help improve future responses."
            )

            feedback = st.text_area(
                "Your feedback:",
                placeholder="e.g., 'Be more detailed', 'Too verbose', 'Perfect!', etc.",
                height=100,
                key="feedback_input",
            )

            col_feedback1, col_feedback2 = st.columns(2)

            with col_feedback1:
                if st.button("Submit Feedback", use_container_width=True):
                    if feedback.strip():
                        # Update system prompt based on feedback
                        updated_prompt = update_system_prompt_from_feedback(
                            feedback, st.session_state.current_system_prompt
                        )
                        st.session_state.current_system_prompt = updated_prompt
                        st.session_state.feedback_text = feedback
                        st.success("✅ Feedback received! System prompt updated.")

                        # Hide feedback form
                        st.session_state.show_feedback = False
                    else:
                        st.warning("Please enter some feedback.")

            with col_feedback2:
                if st.button("Skip Feedback", use_container_width=True):
                    st.session_state.show_feedback = False
                    st.info("Feedback skipped.")

        # Rerun section
        if st.session_state.last_user_input and not st.session_state.show_feedback:
            st.subheader("🔄 Rerun")
            st.info(f"Rerun with updated configuration:")
            st.text(f"Last input: {st.session_state.last_user_input[:50]}...")

            if st.button("Rerun Last Input", use_container_width=True):
                if not client:
                    st.error("Cannot rerun: No connection to LangGraph server")
                else:
                    # Add separator to chat history
                    st.session_state.messages.append(
                        {
                            "role": "system",
                            "content": "--- Rerunning with updated configuration ---",
                        }
                    )

                    # Add user message to chat history
                    st.session_state.messages.append(
                        {"role": "user", "content": st.session_state.last_user_input}
                    )

                    # Stream agent response with updated configuration
                    with st.spinner(
                        "🤖 Agent is thinking with updated configuration..."
                    ):
                        try:
                            response = asyncio.run(
                                stream_agent_response(
                                    client,
                                    st.session_state.last_user_input,
                                    st.session_state.current_system_prompt,
                                )
                            )

                            # Store agent response
                            st.session_state.agent_response = response
                            st.session_state.streaming_complete = True

                            # Add agent response to chat history
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response}
                            )

                            # Show feedback form
                            st.session_state.show_feedback = True

                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.show_feedback = False
                st.session_state.streaming_complete = False
                st.session_state.last_user_input = ""
                st.rerun()


if __name__ == "__main__":
    main()
