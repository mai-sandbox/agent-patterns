# 🔥 Web-Enabled AI Agent

A powerful LangGraph-based AI agent that provides web search capabilities and webpage summarization using Tavily Search Engine and Firecrawl. This agent can perform real-time web searches, summarize webpage content, and engage in natural conversations with users.

## ✨ Features

- **🔍 Web Search**: Real-time web search using Tavily Search Engine with relevant results and summaries
- **📄 Webpage Summarization**: Extract and summarize webpage content using Firecrawl for clean, LLM-ready markdown
- **💬 Conversational Interface**: Natural language interaction with context-aware responses
- **🏗️ LangGraph Architecture**: Built on LangGraph's StateGraph for reliable, scalable agent workflows
- **🛠️ Tool Integration**: Seamless integration of multiple tools with conditional routing
- **⚡ Streaming Support**: Real-time response streaming for better user experience
- **🔧 Production Ready**: Configured for deployment with LangGraph Platform compatibility

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Anthropic Claude](https://console.anthropic.com/) (preferred LLM provider)
  - [Tavily Search](https://tavily.com/) (web search)
  - [Firecrawl](https://firecrawl.dev/) (webpage scraping)

### Installation

1. **Clone the repository** (or download the files):
   ```bash
   git clone <repository-url>
   cd agent-patterns
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Unix/macOS
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your API keys
   nano .env  # or use your preferred editor
   ```

## 🔑 Environment Configuration

Create a `.env` file in the project root with your API keys:

```env
# Required: LLM Provider (Anthropic Claude - preferred)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Required: Web Search (Tavily Search Engine)
TAVILY_API_KEY=your_tavily_api_key_here

# Required: Web Scraping (Firecrawl)
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Optional: Alternative LLM providers
# OPENAI_API_KEY=your_openai_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

# Optional: LangSmith tracing (for debugging)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_api_key_here
# LANGCHAIN_PROJECT=web-agent-project
```

### API Key Setup Instructions

#### 1. Anthropic Claude API Key
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Create a new API key
5. Copy the key and add it to your `.env` file

#### 2. Tavily Search API Key
1. Go to [Tavily](https://tavily.com/)
2. Sign up for an account
3. Navigate to your dashboard
4. Find your API key in the account settings
5. Copy the key and add it to your `.env` file

#### 3. Firecrawl API Key
1. Visit [Firecrawl](https://firecrawl.dev/)
2. Create an account and log in
3. Go to your dashboard
4. Generate an API key
5. Copy the key and add it to your `.env` file

## 💻 Usage

### Command Line Interface

Run the agent interactively from the command line:

```bash
python agent.py
```

This will start an interactive session where you can:
- Ask questions that require web search
- Request webpage summarization
- Have general conversations

### Example Interactions

#### Web Search Example
```
🧑 You: What are the latest developments in AI technology?

🤖 Assistant: I'll search for the latest AI technology developments for you.

[Performs web search using Tavily]

Based on my search, here are the latest developments in AI technology:

1. **Generative AI Advances**: Recent breakthroughs in large language models...
2. **AI Hardware**: New developments in AI chips and computing infrastructure...
[Detailed response with current information]
```

#### Webpage Summarization Example
```
🧑 You: Can you summarize this article for me? https://example.com/ai-article

🤖 Assistant: I'll extract and summarize the content from that webpage for you.

[Uses Firecrawl to scrape and process the webpage]

**Webpage Summary**
**URL:** https://example.com/ai-article
**Title:** Latest AI Research Findings

**Key Points:**
- Main finding 1: [Summary of key point]
- Main finding 2: [Summary of key point]
[Detailed summary of the webpage content]
```

#### General Conversation Example
```
🧑 You: How do neural networks work?

🤖 Assistant: Neural networks are computational models inspired by biological neural networks...
[Provides detailed explanation without needing external tools]
```

### Programmatic Usage

You can also use the agent programmatically:

```python
from agent import app

# Single interaction
result = app.invoke({
    "messages": [{"role": "user", "content": "Search for recent AI news"}]
})

# Streaming interaction
for event in app.stream({
    "messages": [{"role": "user", "content": "Summarize https://example.com"}]
}):
    for value in event.values():
        if "messages" in value:
            print(value["messages"][-1].content)
```

### LangGraph Platform Deployment

The agent is configured for deployment on LangGraph Platform:

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Start local development server
langgraph dev

# Deploy to LangGraph Platform
langgraph deploy
```

## 🏗️ Architecture

The agent is built using LangGraph's StateGraph architecture:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│    START    │───▶│   Chatbot    │───▶│    END      │
└─────────────┘    └──────────────┘    └─────────────┘
                           │                    ▲
                           ▼                    │
                   ┌──────────────┐            │
                   │    Tools     │────────────┘
                   └──────────────┘
```

### Components

- **State Management**: Uses TypedDict with `add_messages` for conversation history
- **Chatbot Node**: Processes user input and decides whether to use tools
- **Tool Node**: Executes web search or webpage summarization tools
- **Conditional Routing**: Automatically routes between conversation and tool usage

### Tools

1. **Web Search Tool** (`web_search`):
   - Uses Tavily Search Engine
   - Returns formatted results with titles, URLs, and summaries
   - Handles search errors gracefully

2. **Webpage Summarization Tool** (`summarize_webpage`):
   - Uses Firecrawl for webpage scraping
   - Extracts clean markdown content
   - Includes metadata (title, description)
   - Validates URL format

## 🔧 Development

### Code Quality

The project includes code quality tools:

```bash
# Run linting
ruff check .

# Run type checking
mypy . --ignore-missing-imports

# Fix linting issues automatically
ruff check . --fix
```

### Project Structure

```
agent-patterns/
├── agent.py              # Main LangGraph agent implementation
├── tools.py              # Web search and summarization tools
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── langgraph.json       # LangGraph platform configuration
├── README.md            # This documentation
└── .gitignore          # Git ignore patterns
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. API Key Errors

**Problem**: `AuthenticationError` or `Invalid API key`

**Solutions**:
- Verify your API keys are correctly set in the `.env` file
- Ensure there are no extra spaces or quotes around the keys
- Check that your API keys are active and have sufficient credits
- Restart the application after updating environment variables

#### 2. Import Errors

**Problem**: `ModuleNotFoundError` for langchain or langgraph packages

**Solutions**:
- Ensure you've activated your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)
- Try upgrading pip: `pip install --upgrade pip`

#### 3. Tool Execution Failures

**Problem**: Web search or webpage summarization not working

**Solutions**:
- **Web Search Issues**:
  - Verify Tavily API key is valid
  - Check internet connectivity
  - Try simpler search queries
  
- **Webpage Summarization Issues**:
  - Ensure the URL is accessible and starts with `http://` or `https://`
  - Check if the website blocks scraping (some sites have anti-bot measures)
  - Verify Firecrawl API key and account status

#### 4. LangGraph Compilation Errors

**Problem**: Graph fails to compile or run

**Solutions**:
- Check that all required dependencies are installed
- Verify the StateGraph structure in `agent.py`
- Ensure tools are properly imported from `tools.py`
- Check for syntax errors in Python files

#### 5. Environment Variable Issues

**Problem**: Environment variables not loading

**Solutions**:
- Ensure `.env` file is in the project root directory
- Check file permissions on `.env` file
- Verify `python-dotenv` is installed
- Try loading environment variables manually:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  import os
  print(os.getenv('ANTHROPIC_API_KEY'))  # Should not be None
  ```

#### 6. Performance Issues

**Problem**: Slow response times or timeouts

**Solutions**:
- Check your internet connection
- Verify API service status (Anthropic, Tavily, Firecrawl)
- Consider reducing the number of search results in `tools.py`
- Monitor API rate limits and usage quotas

#### 7. Memory or Resource Issues

**Problem**: High memory usage or crashes

**Solutions**:
- Ensure you have sufficient RAM (recommended: 4GB+)
- Close other resource-intensive applications
- Consider using a smaller language model if available
- Monitor system resources during execution

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Run the agent with verbose logging to see detailed error messages
2. **Verify dependencies**: Ensure all packages are up to date
3. **Test components individually**: Try running tools separately to isolate issues
4. **Check API status**: Verify that external services (Anthropic, Tavily, Firecrawl) are operational
5. **Review documentation**: Check the official LangGraph and LangChain documentation for updates

### Debug Mode

For debugging, you can enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use LangSmith for detailed tracing by setting the optional environment variables in your `.env` file.

## 📝 License

This project is provided as-is for educational and development purposes. Please ensure you comply with the terms of service for all external APIs used (Anthropic, Tavily, Firecrawl).

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this agent implementation.

---

**Happy coding! 🚀**
