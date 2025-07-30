# 🤖 LangGraph Agent with Streamlit Feedback Interface

A comprehensive Streamlit application that connects to a LangGraph agent for interactive chat with real-time feedback and configuration updates. The agent includes web search capabilities via Tavily and supports dynamic system prompt configuration based on user feedback.

## 🌟 Features

- **Interactive Chat Interface**: Clean, intuitive Streamlit UI for chatting with the LangGraph agent
- **Real-time Streaming**: Live streaming of agent responses as they're generated
- **Tool Calling**: Integrated Tavily web search for current information retrieval
- **Feedback Loop**: Collect user feedback and automatically update agent configuration
- **Dynamic Configuration**: System prompt updates based on user feedback
- **Rerun Capability**: Test improved responses with updated configuration
- **Connection Monitoring**: Visual indicators for LangGraph server connection status

## 🏗️ Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌──────────────────┐
│   Streamlit     │ ◄─────────────────► │   LangGraph      │
│   Frontend      │                     │   Agent Server   │
│   (app.py)      │                     │   (agent.py)     │
└─────────────────┘                     └──────────────────┘
         │                                        │
         │                                        │
         ▼                                        ▼
┌─────────────────┐                     ┌──────────────────┐
│   User          │                     │   Tavily Search  │
│   Feedback      │                     │   Tool           │
│   Processing    │                     │   (Web Search)   │
└─────────────────┘                     └──────────────────┘
```

## 📋 Prerequisites

Before you begin, ensure you have the following:

- **Python 3.11+** (Required for LangGraph)
- **API Keys**:
  - [Anthropic API Key](https://console.anthropic.com/settings/keys) (Preferred LLM)
  - [Tavily API Key](https://app.tavily.com/) (Web search functionality)
  - [LangSmith API Key](https://smith.langchain.com/settings) (Optional, for tracing)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd agent-patterns

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual API keys
nano .env  # or use your preferred editor
```

**Required environment variables:**
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGGRAPH_API_URL=http://localhost:2024
```

### 4. Start the LangGraph Agent Server

```bash
# Start the LangGraph development server
langgraph dev
```

You should see output similar to:
```
>    Ready!
>
>    - API: http://localhost:2024/
>    - Docs: http://localhost:2024/docs
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 5. Launch the Streamlit Application

In a **new terminal window**:

```bash
# Activate the virtual environment (if not already active)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the Streamlit app
streamlit run app.py
```

### 6. Access the Application

Open your browser and navigate to: **http://localhost:8501**

## 📖 Usage Guide

### Basic Chat

1. **Enter your message** in the text area on the main chat interface
2. **Click "Send Message"** to submit your query
3. **Watch the real-time response** stream in from the agent
4. The agent will automatically use web search when needed for current information

### System Prompt Configuration

1. **Open the sidebar** (⚙️ Configuration)
2. **Modify the system prompt** in the text area
3. **Changes apply immediately** to new conversations
4. The default prompt encourages web search for current information

### Feedback and Improvement Loop

1. **After receiving an agent response**, a feedback section appears
2. **Enter your feedback** about the response quality
3. **Click "Submit Feedback"** to update the system prompt
4. **Use "Rerun Previous Input"** to test the improved configuration
5. **Compare responses** to see the improvement

### Example Feedback Types

- **"Be more detailed"** → Adds instruction for comprehensive explanations
- **"Be more concise"** → Adds instruction for brief responses
- **"Include more examples"** → Adds instruction to provide examples
- **"Use more formal tone"** → Adds professional tone instruction
- **"Search more often"** → Emphasizes using the search tool

## 🔧 Configuration Options

### System Prompt Customization

The system prompt can be customized in several ways:

1. **Via Streamlit UI**: Use the sidebar configuration panel
2. **Via Feedback**: Automatic updates based on user feedback
3. **Via Code**: Modify the default in `app.py`

### LangGraph Agent Configuration

The agent configuration is defined in `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:app"
  },
  "env": "./.env"
}
```

### Tavily Search Configuration

Search tool settings in `agent.py`:

```python
search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
)
```

## 🛠️ Development

### Project Structure

```
agent-patterns/
├── agent.py              # LangGraph agent implementation
├── app.py                # Streamlit application
├── langgraph.json        # LangGraph configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
├── .env                 # Your environment variables (not in git)
├── README.md            # This file
└── Agent.md             # Development guidelines
```

### Key Components

- **`agent.py`**: Contains the LangGraph StateGraph with chatbot and tool nodes
- **`app.py`**: Streamlit interface with streaming, feedback, and configuration
- **`langgraph.json`**: Configuration for LangGraph deployment
- **`requirements.txt`**: All necessary Python dependencies

### Running Tests

```bash
# Install development dependencies
pip install ruff mypy

# Run code quality checks
ruff check .
mypy . --ignore-missing-imports
```

## 🐛 Troubleshooting

### Common Issues

#### 1. LangGraph Server Connection Failed

**Error**: `Failed to connect to LangGraph server at http://localhost:2024`

**Solutions**:
- Ensure `langgraph dev` is running in another terminal
- Check that port 2024 is not blocked by firewall
- Verify the `LANGGRAPH_API_URL` in your `.env` file

#### 2. API Key Errors

**Error**: `AuthenticationError` or `Invalid API key`

**Solutions**:
- Verify API keys are correctly set in `.env` file
- Ensure no extra spaces or quotes around API keys
- Check that API keys are valid and have sufficient credits

#### 3. Import Errors

**Error**: `ModuleNotFoundError: No module named 'langgraph'`

**Solutions**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Verify Python version is 3.11+

#### 4. Streaming Issues

**Error**: Responses not streaming or appearing slowly

**Solutions**:
- Check network connection to LangGraph server
- Verify `nest_asyncio.apply()` is called in `app.py`
- Restart both LangGraph server and Streamlit app

#### 5. Feedback Not Updating Configuration

**Issue**: System prompt not changing after feedback

**Solutions**:
- Check the feedback processing logic in `process_feedback_and_update_config()`
- Verify the sidebar shows the updated system prompt
- Try using more specific feedback keywords

### Debug Mode

Enable debug information in the Streamlit app:

1. **Expand the "🔍 Debug Information" section** in the right panel
2. **Check current state values** for troubleshooting
3. **Verify configuration updates** are being applied

### Logs and Monitoring

- **LangGraph Server Logs**: Check the terminal running `langgraph dev`
- **Streamlit Logs**: Check the terminal running `streamlit run app.py`
- **LangSmith Tracing**: Enable in `.env` for detailed execution traces

## 🔍 Advanced Usage

### Custom Tool Integration

To add new tools to the agent:

1. **Import the tool** in `agent.py`
2. **Add to the tools list**
3. **Update the LLM binding**: `llm_with_tools = llm.bind_tools(tools)`

### Custom Feedback Processing

Modify `process_feedback_and_update_config()` in `app.py` to add custom feedback rules:

```python
def process_feedback_and_update_config(feedback: str, current_prompt: str) -> str:
    # Add your custom feedback processing logic here
    if "your_custom_keyword" in feedback.lower():
        return current_prompt + " Your custom instruction."
    return current_prompt
```

### Multi-turn Conversations

To enable conversation memory:

1. **Create a thread** instead of using stateless runs
2. **Store thread_id** in session state
3. **Pass thread_id** to `client.runs.stream()`

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Platform Guide](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Tavily Search API](https://docs.tavily.com/)

## 🤝 Contributing

1. **Follow the guidelines** in `Agent.md`
2. **Run code quality checks** before submitting
3. **Test both LangGraph agent and Streamlit interface**
4. **Update documentation** for any new features

## 📄 License

This project is provided as-is for educational and development purposes.

---

**Need help?** Check the troubleshooting section above or create an issue with detailed error information.
