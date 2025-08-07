# Interactive LangGraph Agent Deployment

A complete system for deploying and interacting with LangGraph agents through a Streamlit web interface. This project demonstrates how to create a configurable chat agent with tool calling capabilities and deploy it locally with real-time streaming responses.

## Features

- **🤖 Intelligent Chat Agent**: Built with LangGraph's `create_react_agent` using Claude 3.5 Sonnet
- **🛠️ Tool Calling**: Includes weather, calculator, and time retrieval tools
- **⚙️ Configurable System Prompt**: Dynamic system prompt updates through the web interface
- **📡 Real-time Streaming**: Live streaming of agent responses using LangGraph SDK
- **💬 Interactive Web UI**: Clean Streamlit interface for seamless interaction
- **🔄 Feedback Loop**: Collect feedback and automatically update agent configuration
- **🚀 Local Development**: Easy local deployment with `langgraph dev`

## Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌──────────────────┐
│   Streamlit     │ ◄─────────────────► │   LangGraph      │
│   Web App       │                     │   Dev Server     │
│                 │                     │   (Port 2024)    │
└─────────────────┘                     └──────────────────┘
         │                                        │
         │                                        │
         ▼                                        ▼
┌─────────────────┐                     ┌──────────────────┐
│  langgraph-sdk  │                     │    agent.py      │
│     Client      │                     │  (Chat Agent)    │
└─────────────────┘                     └──────────────────┘
```

## Prerequisites

- **Python 3.11+** (Python 3.13.3 recommended)
- **API Keys**: You'll need at least one of the following:
  - `ANTHROPIC_API_KEY` (recommended - for Claude models)
  - `OPENAI_API_KEY` (for GPT models)
  - `GOOGLE_API_KEY` (for Gemini models)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- `streamlit` - Web interface framework
- `langgraph` - Core LangGraph framework
- `langgraph-cli[inmem]` - CLI with in-memory support
- `langgraph-sdk` - SDK for client communication
- `langchain-anthropic` - Anthropic model integration
- `nest-asyncio` - Async support for Streamlit

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Required: At least one API key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

# Optional: LangSmith for tracing (recommended)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
```

### 3. Start the LangGraph Development Server

```bash
langgraph dev
```

You should see output like:
```
>    Ready!
>
>    - API: http://localhost:2024/
>    - Docs: http://localhost:2024/docs
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 4. Run the Streamlit App

In a new terminal window:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Examples

### Basic Chat Interaction

1. **Start a conversation**: Type a message in the input field
2. **Watch real-time responses**: See the agent's response stream in real-time
3. **Use tools**: Ask questions that trigger tool usage:
   - "What's the weather in San Francisco?"
   - "Calculate 15 * 23 + 7"
   - "What time is it?"

### System Prompt Customization

1. **Modify behavior**: Use the system prompt field to customize the agent:
   ```
   You are a helpful coding assistant. Always provide code examples and explain your reasoning step by step.
   ```

2. **Apply changes**: The new system prompt takes effect immediately for new conversations

### Feedback and Improvement

1. **Provide feedback**: After each response, use the feedback form to rate the agent's performance
2. **Automatic updates**: The system uses feedback to suggest improved system prompts
3. **Iterative improvement**: Continue the feedback loop to refine agent behavior

## Project Structure

```
agent-patterns/
├── agent.py              # LangGraph chat agent implementation
├── langgraph.json        # LangGraph server configuration
├── streamlit_app.py      # Streamlit web interface
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .env                 # Environment variables (create this)
```

## Configuration Files

### `agent.py`
- Implements the chat agent using `create_react_agent`
- Includes weather, calculator, and time tools
- Supports configurable system prompts
- Exports the compiled graph as `app` for deployment

### `langgraph.json`
- Defines the assistant configuration for the LangGraph server
- Specifies configurable parameters (system_prompt)
- Points to the agent implementation

### `streamlit_app.py`
- Provides the web interface for agent interaction
- Handles real-time streaming using LangGraph SDK
- Manages configuration updates and feedback collection
- Includes error handling and connection management

## Advanced Usage

### Custom Tools

Add new tools to `agent.py`:

```python
@tool
def my_custom_tool(input_text: str) -> str:
    """Description of what this tool does."""
    # Your tool implementation
    return result

# Add to tools list
tools = [get_weather, calculator, get_current_time, my_custom_tool]
```

### Different Models

Modify the model in `agent.py`:

```python
# Use OpenAI instead of Anthropic
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

# Use Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
```

### Production Deployment

For production deployment:

1. **Replace in-memory storage**: Update `langgraph.json` to use persistent storage
2. **Environment management**: Use proper environment variable management
3. **Security**: Add authentication and rate limiting
4. **Scaling**: Consider using LangGraph Platform for cloud deployment

## Troubleshooting

### Common Issues

**1. "Connection refused" error**
- Ensure the LangGraph dev server is running (`langgraph dev`)
- Check that the server is accessible at `http://localhost:2024`

**2. "API key not found" error**
- Verify your API keys are set in the `.env` file
- Ensure the `.env` file is in the project root directory

**3. Streamlit async issues**
- The app uses `nest-asyncio` to handle async operations
- If you encounter event loop errors, restart the Streamlit app

**4. Tool calling not working**
- Check that your model supports tool calling (Claude 3.5 Sonnet does)
- Verify tool definitions are properly formatted

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export LANGCHAIN_VERBOSE=true
export LANGCHAIN_DEBUG=true
```

## Development

### Testing the Agent

Test the agent directly without the web interface:

```python
from agent import app

# Test basic functionality
result = app.invoke({"messages": [{"role": "human", "content": "Hello!"}]})
print(result)
```

### LangGraph Studio

Access the visual debugging interface at:
```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangGraph Platform](https://docs.langchain.com/langgraph-platform)
- [LangSmith](https://smith.langchain.com/) - For tracing and monitoring
