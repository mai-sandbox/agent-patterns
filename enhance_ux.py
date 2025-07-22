#!/usr/bin/env python3
import re

# Read the current app.py file
with open('app.py', 'r') as f:
    content = f.read()

# Add enhanced error handling to the streaming function
# Replace the basic error handling with more comprehensive error handling
old_error_handling = r'except Exception as e:\s+st\.error\(f"Error streaming agent response: \{str\(e\)\}"\)\s+return None, None'

new_error_handling = '''except Exception as e:
        # Enhanced error handling with more specific error messages
        error_msg = str(e)
        if "connection" in error_msg.lower():
            st.error("🔌 **Connection Error**: Unable to connect to the LangGraph agent. Please check if the agent is running and the URL is correct.")
        elif "timeout" in error_msg.lower():
            st.error("⏱️ **Timeout Error**: The agent took too long to respond. Please try again or check the agent's performance.")
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            st.error("🔐 **Authentication Error**: Access denied. Please check your credentials or agent configuration.")
        elif "not found" in error_msg.lower() or "404" in error_msg:
            st.error("🔍 **Agent Not Found**: The specified agent name was not found. Please verify the agent name in your configuration.")
        else:
            st.error(f"❌ **Unexpected Error**: {error_msg}")
        
        # Provide troubleshooting suggestions
        with st.expander("🛠️ Troubleshooting Tips"):
            st.markdown("""
            **Common solutions:**
            1. **Check agent status**: Ensure your LangGraph agent is running with `langgraph dev`
            2. **Verify URL**: Confirm the agent URL is correct (default: http://localhost:2024)
            3. **Check agent name**: Verify the agent name matches your `langgraph.json` configuration
            4. **Network issues**: Try refreshing the page or restarting the agent
            5. **Configuration**: Review your JSON configuration for syntax errors
            """)
        
        return None, None'''

content = re.sub(old_error_handling, new_error_handling, content, flags=re.DOTALL)

# Add progress indicators to the streaming function
# Add progress tracking after the thread creation
thread_creation = r'thread = client\.threads\.create\(\)'
enhanced_thread_creation = '''# Enhanced progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Create thread
        status_text.text("🔄 Creating conversation thread...")
        progress_bar.progress(10)
        
        thread = client.threads.create()
        
        # Step 2: Configure thread
        status_text.text("⚙️ Configuring agent settings...")
        progress_bar.progress(20)'''

content = re.sub(thread_creation, enhanced_thread_creation, content)

# Add streaming progress updates
streaming_start = r'# Stream the agent response\s+response_container = st\.empty\(\)\s+full_response = ""'
enhanced_streaming_start = '''# Step 3: Initialize streaming
        status_text.text("📡 Starting agent execution...")
        progress_bar.progress(30)
        
        # Stream the agent response
        response_container = st.empty()
        full_response = ""
        step_count = 0
        total_chunks = 0'''

content = re.sub(streaming_start, enhanced_streaming_start, content, flags=re.DOTALL)

# Add completion indicators
return_statement = r'return full_response, thread\["thread_id"\]'
enhanced_return = '''# Step 4: Completion
        status_text.text("✅ Agent execution completed!")
        progress_bar.progress(100)
        
        # Clear progress indicators after a short delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return full_response, thread["thread_id"]'''

content = re.sub(return_statement, enhanced_return, content)

# Write the modified content back
with open('app.py', 'w') as f:
    f.write(content)

print('Successfully enhanced UX with better error handling and progress indicators')

