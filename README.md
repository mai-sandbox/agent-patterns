# 🤖 LangGraph Agent Interaction & Feedback App

A Streamlit application for interacting with LangGraph agents deployed locally, featuring real-time streaming, feedback collection, and dynamic configuration updates.

## ✨ Features

- **🔗 LangGraph Integration**: Connect to locally deployed LangGraph agents via `langgraph dev`
- **📡 Real-time Streaming**: Stream agent execution progress in real-time
- **💭 Feedback Collection**: Provide feedback on agent performance after each run
- **⚙️ Dynamic Configuration**: Update agent configuration based on user feedback
- **🔄 Rerun Capability**: Re-execute agents with updated configurations
- **📝 Thread Management**: Persistent conversation threads for context continuity
- **🎛️ Custom Configuration**: JSON-based configuration editing
- **📊 Status Monitoring**: Real-time connection and execution status indicators

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A LangGraph agent project ready for deployment
- Git (for cloning the repository)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt
```

### 2. Start LangGraph Development Server

Before running the Streamlit app, you need to start your LangGraph agent server:

```bash
# Navigate to your LangGraph agent project directory
cd /path/to/your/langgraph/agent

# Start the development server (this will run on http://localhost:2024 by default)
langgraph dev
```

**Important**: Keep this terminal window open as the server needs to remain running.

### 3. Launch Streamlit App

In a new terminal window:

```bash
# Navigate back to this project directory
cd /path/to/this/streamlit/app

# Launch the Streamlit application
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## 📋 Usage Instructions

### Step 1: Connect to LangGraph Server

1. Open the Streamlit app in your browser
2. In the sidebar, verify the server URL is `http://localhost:2024`
3. Click **"Connect to LangGraph Server"**
4. Wait for the success message confirming connection and thread creation

### Step 2: Configure Your Agent

1. Set the **Assistant ID** (default: "agent")
2. Optionally modify the **Custom Configuration** JSON
3. The configuration will be applied to your agent runs

### Step 3: Run Your Agent

1. Enter your message in the **Input** text area
2. Click **"🚀 Run Agent"**
3. Watch the real-time streaming updates in the **Agent Progress** section
4. Wait for the run to complete

### Step 4: Provide Feedback

1. After the agent run completes, the **Feedback & Configuration Update** section will appear
2. Review the **Agent Response Summary**
3. Enter your feedback in the text area
4. Click **"📝 Update Configuration"** to incorporate your feedback
5. Click **"🔄 Rerun with Updated Config"** to run the agent again with improvements

## 🔧 Configuration

### Environment Variables

You can create a `.env` file in the project root to set environment variables:

```env
LANGGRAPH_SERVER_URL=http://localhost:2024
DEFAULT_ASSISTANT_ID=agent
```

### Custom Agent Configuration

The app supports custom JSON configuration that gets passed to your LangGraph agent. Example configurations:

```json
{
  "configurable": {
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

## 🛠️ Troubleshooting

### Connection Issues

**Problem**: "Failed to connect to LangGraph server"

**Solutions**:
- Ensure your LangGraph server is running with `langgraph dev`
- Verify the server URL is correct (default: `http://localhost:2024`)
- Check that no firewall is blocking the connection
- Ensure your LangGraph agent project has a valid `langgraph.json` configuration

### Agent Run Failures

**Problem**: Agent runs fail or don't stream properly

**Solutions**:
- Check your LangGraph agent logs in the terminal running `langgraph dev`
- Verify your agent configuration is valid JSON
- Ensure your agent has the correct assistant ID
- Try with a simpler input message first

### Streaming Issues

**Problem**: No streaming updates appear

**Solutions**:
- Refresh the Streamlit app
- Check browser console for JavaScript errors
- Verify the LangGraph server is responding to requests
- Try reconnecting to the server

## 📦 Dependencies

- **streamlit>=1.28.0**: Web application framework
- **langgraph-sdk>=0.1.0**: LangGraph client SDK
- **requests>=2.31.0**: HTTP client library
- **python-dotenv>=1.0.0**: Environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**💡 Pro Tip**: Keep your LangGraph development server running in a separate terminal while using this app for the best experience!

