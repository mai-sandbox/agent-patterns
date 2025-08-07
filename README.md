# LangGraph Agent Interaction Streamlit App

A comprehensive Streamlit application for interacting with LangGraph agents, featuring real-time streaming responses, configurable system prompts, and an intelligent feedback collection system.

## 🚀 Features

- **Interactive Chat Interface**: Clean, intuitive chat interface for agent interaction
- **Real-time Streaming**: Live streaming of agent responses with progress indicators
- **Configurable System Prompts**: Dynamic system prompt editing with immediate updates
- **Tool Calling Support**: Agent equipped with calculator and web search simulation tools
- **Feedback Collection**: Intelligent feedback system that updates agent behavior
- **Session Management**: Persistent conversation history and state management
- **Error Handling**: Robust error handling with fallback mechanisms

## 📋 Prerequisites

- Python 3.9 or higher
- API key for your chosen LLM provider (OpenAI, Anthropic, etc.)
- Git (for cloning the repository)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agent-patterns
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file with your actual API keys:

```env
# Required: Choose your LLM provider
OPENAI_API_KEY=your_openai_api_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LangGraph Server (default is usually fine)
LANGGRAPH_SERVER_URL=http://localhost:8123
```

## 🚀 Quick Start

### Step 1: Start the LangGraph Agent Server

In your first terminal, start the LangGraph development server:

```bash
langgraph dev
```

This will:
- Start the agent server on `http://localhost:8123`
- Load the agent configuration from `langgraph.json`
- Make the agent available for API calls

**Expected Output:**
```
Starting LangGraph server...
Server running on http://localhost:8123
Agent loaded successfully
```

### Step 2: Launch the Streamlit App

In a second terminal (keep the first one running), start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

This will:
- Launch the Streamlit web interface
- Connect to the LangGraph server
- Open your browser to `http://localhost:8501`

**Expected Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

### Step 3: Start Chatting!

1. Open your browser to `http://localhost:8501`
2. Configure the system prompt in the sidebar (optional)
3. Type your message in the chat input
4. Watch the agent respond in real-time with streaming
5. Provide feedback to improve future responses

## 📖 Usage Guide

### Basic Chat Interaction

1. **Send Messages**: Type your message in the chat input at the bottom
2. **View Responses**: Agent responses stream in real-time with progress indicators
3. **Review History**: All messages are preserved in the conversation history

### System Prompt Configuration

1. **Access Configuration**: Use the sidebar "🔧 Configuration" panel
2. **Edit Prompt**: Modify the system prompt in the text area
3. **Apply Changes**: Changes are applied immediately to new conversations
4. **Reset**: Clear conversation to start fresh with new prompt

### Feedback System

After each agent response, you can:

1. **Provide Feedback**: Use the feedback text area that appears
2. **Submit Feedback**: Click "📝 Submit Feedback" to update the system prompt
3. **Rerun with Updates**: Click "🔄 Rerun with Updated Config" to test changes
4. **View Changes**: Expand the system prompt changes to see what was modified

### Streaming Controls

- **Enable/Disable Streaming**: Toggle in the sidebar configuration
- **Progress Indicators**: Watch real-time progress during agent execution
- **Fallback Mode**: Automatic fallback to non-streaming if issues occur

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LANGGRAPH_SERVER_URL` | LangGraph server endpoint | `http://localhost:8123` | No |
| `OPENAI_API_KEY` | OpenAI API key | None | Yes* |
| `ANTHROPIC_API_KEY` | Anthropic API key | None | Yes* |
| `ENVIRONMENT` | Environment mode | `development` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

*At least one LLM provider API key is required

### LangGraph Configuration

The `langgraph.json` file controls:
- Agent entry point (`agent.py`)
- Environment variables
- Development server settings
- Python version requirements

### Agent Tools

The agent comes with built-in tools:
- **Calculator**: Perform mathematical calculations
- **Web Search Simulator**: Simulate web search results
- **Custom Tools**: Easily extensible for additional functionality

## 🐛 Troubleshooting

### Common Issues

#### 1. "Connection failed" Error

**Problem**: Streamlit app can't connect to LangGraph server

**Solutions**:
- Ensure LangGraph server is running (`langgraph dev`)
- Check server URL in `.env` file
- Verify server is accessible at `http://localhost:8123`
- Check firewall settings

**Debug Steps**:
```bash
# Test server connectivity
curl http://localhost:8123/health

# Check server logs
# Look at the terminal running `langgraph dev`
```

#### 2. "API Key Not Found" Error

**Problem**: Missing or invalid LLM provider API key

**Solutions**:
- Verify API key is set in `.env` file
- Check API key format and validity
- Ensure you're using the correct provider key

**Debug Steps**:
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

#### 3. Streaming Not Working

**Problem**: Responses not streaming in real-time

**Solutions**:
- Check "Enable Streaming" in sidebar
- Verify LangGraph server supports streaming
- Try disabling/re-enabling streaming
- Check browser console for errors

#### 4. Feedback System Not Updating

**Problem**: Feedback not updating system prompt

**Solutions**:
- Ensure feedback text is not empty
- Check that "Submit Feedback" was clicked
- Verify system prompt changes in sidebar
- Try clearing conversation and starting fresh

#### 5. Import Errors

**Problem**: Missing dependencies or import failures

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.9+

# Verify virtual environment
which python
```

### Performance Issues

#### Slow Response Times

- Check internet connection for API calls
- Verify LLM provider service status
- Consider using a different model
- Check system resources (CPU, memory)

#### Memory Usage

- Clear conversation history regularly
- Restart Streamlit app if memory grows
- Monitor system resources

### Development Issues

#### Code Quality Checks

```bash
# Run linting
ruff check .

# Run type checking
mypy . --ignore-missing-imports

# Fix common issues
ruff check . --fix
```

## 📚 Advanced Usage

### Custom System Prompts

Create specialized agents by modifying the system prompt:

```
You are a Python programming expert. Help users with:
- Code debugging and optimization
- Best practices and patterns
- Library recommendations
- Performance improvements

Always provide working code examples.
```

### Adding Custom Tools

Extend the agent with custom tools by modifying `agent.py`:

```python
@tool
def custom_tool(query: str) -> str:
    """Your custom tool description."""
    # Implementation here
    return result
```

### Environment-Specific Configuration

Use different configurations for development and production:

```env
# Development
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Production
ENVIRONMENT=production
LOG_LEVEL=INFO
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run quality checks: `ruff check .` and `mypy .`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review the [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
3. Check [Streamlit documentation](https://docs.streamlit.io/)
4. Open an issue on GitHub

## 🔄 Version History

- **v1.0.0**: Initial release with basic chat functionality
- **v1.1.0**: Added streaming support
- **v1.2.0**: Implemented feedback collection system
- **v1.3.0**: Enhanced error handling and documentation

---

**Happy chatting with your LangGraph agent! 🤖✨**
