# 🤖 LangGraph Chatbot

A comprehensive chatbot implementation built with LangGraph that supports multiple LLM providers, proper state management, and an interactive command-line interface.

## 🌟 Features

- **Multi-Provider LLM Support**: Compatible with OpenAI, Anthropic, and Google Gemini
- **LangGraph Architecture**: Built using LangGraph's state machine approach for robust conversation flow
- **Interactive CLI**: User-friendly command-line interface with streaming responses
- **Automatic Provider Detection**: Automatically selects the first available LLM provider
- **Type-Safe Implementation**: Comprehensive type hints throughout the codebase
- **Error Handling**: Robust error handling with custom exceptions and user-friendly messages
- **Environment Configuration**: Secure API key management using environment variables
- **Conversation Memory**: Maintains conversation context using LangGraph's state management

## 🏗️ Project Structure

```
agent-patterns/
├── chatbot.py          # Main chatbot implementation
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variable template
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- API key for at least one supported LLM provider

## 🚀 Installation

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd agent-patterns

# Or download and extract the project files
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔧 Environment Setup

### 1. Create Environment File

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit the `.env` file and add your API key for at least one provider:

```bash
# Choose ONE of the following providers:

# Option 1: OpenAI (GPT-4o-mini)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Option 2: Anthropic (Claude-3.5-Sonnet)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Option 3: Google (Gemini-2.0-Flash)
GOOGLE_API_KEY=your-google-api-key-here

# Optional: LangSmith for tracing (recommended for debugging)
LANGSMITH_API_KEY=your-langsmith-api-key-here
```

### 3. Obtain API Keys

#### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

#### Anthropic
1. Visit [Anthropic Console](https://console.anthropic.com/settings/keys)
2. Sign up or log in to your account
3. Go to Settings → API Keys
4. Create a new API key
5. Copy the key to your `.env` file

#### Google Gemini
1. Visit [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
2. Sign up or log in to your Google account
3. Create a new API key
4. Copy the key to your `.env` file

#### LangSmith (Optional)
1. Visit [LangSmith](https://smith.langchain.com/)
2. Sign up for an account
3. Create an API key in your settings
4. Copy the key to your `.env` file

## 🎯 Usage

### Basic Usage

Run the chatbot with:

```bash
python chatbot.py
```

### Interactive Commands

Once the chatbot is running:

- **Chat**: Type any message and press Enter
- **Exit**: Type `quit`, `exit`, or `q` to end the conversation
- **Interrupt**: Press `Ctrl+C` to exit at any time

### Example Conversation

```
🤖 LangGraph Chatbot
============================================================
Using LLM Provider: Anthropic

Instructions:
- Type your message and press Enter to chat
- Type 'quit', 'exit', or 'q' to end the conversation
- The chatbot will remember the conversation context
============================================================

Initializing chatbot...
Chatbot ready! Start chatting below.

You: Hello! What is LangGraph?
Assistant: LangGraph is a library for building stateful, multi-actor applications with LLMs. It's designed to help you create complex workflows and state machines that coordinate multiple AI agents or language model interactions...

You: Can you give me an example?
Assistant: Certainly! Here's a simple example of how LangGraph works...

You: quit
👋 Goodbye! Thanks for chatting!
```

## 🔧 Supported LLM Providers

The chatbot automatically detects and uses the first available provider in this order:

### 1. OpenAI
- **Model**: GPT-4o-mini
- **Environment Variable**: `OPENAI_API_KEY`
- **Features**: Fast responses, good general knowledge
- **Cost**: Pay-per-token pricing

### 2. Anthropic
- **Model**: Claude-3.5-Sonnet
- **Environment Variable**: `ANTHROPIC_API_KEY`
- **Features**: Excellent reasoning, safety-focused
- **Cost**: Pay-per-token pricing

### 3. Google Gemini
- **Model**: Gemini-2.0-Flash
- **Environment Variable**: `GOOGLE_API_KEY`
- **Features**: Fast inference, multimodal capabilities
- **Cost**: Generous free tier available

## 🛠️ Development

### Code Quality

The project follows strict code quality standards:

```bash
# Type checking (recommended)
mypy chatbot.py --ignore-missing-imports

# Code formatting (if available)
ruff check .
```

### Project Architecture

The chatbot is built using LangGraph's state machine approach:

1. **State Management**: Uses `TypedDict` with `add_messages` reducer
2. **Graph Structure**: Simple START → chatbot → END flow
3. **Node Function**: Single `chatbot_node` that processes messages
4. **Provider Abstraction**: Flexible LLM provider initialization
5. **Error Handling**: Custom exceptions and graceful error recovery

### Key Components

- **`State` class**: Defines the conversation state schema
- **`chatbot_node()` function**: Core message processing logic
- **`create_chatbot_graph()`**: Builds and compiles the LangGraph workflow
- **`stream_graph_updates()`**: Handles streaming responses
- **Provider detection**: Automatic LLM provider selection

## 🐛 Troubleshooting

### Common Issues

#### 1. "No LLM provider configured" Error

**Problem**: No API keys are set in environment variables.

**Solution**:
```bash
# Check your .env file exists and contains at least one API key
cat .env

# Ensure the .env file is in the same directory as chatbot.py
ls -la .env

# Verify environment variables are loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENAI_API_KEY:', bool(os.getenv('OPENAI_API_KEY')))"
```

#### 2. "Failed to initialize [provider] LLM" Error

**Problem**: Invalid API key or network issues.

**Solutions**:
- Verify your API key is correct and active
- Check your internet connection
- Ensure you have sufficient credits/quota with the provider
- Try a different provider by setting a different API key

#### 3. Import Errors

**Problem**: Missing dependencies.

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check if packages are installed
pip list | grep -E "(langgraph|langchain)"
```

#### 4. Permission Errors

**Problem**: Cannot read .env file or execute chatbot.py.

**Solution**:
```bash
# Fix file permissions
chmod 644 .env
chmod 755 chatbot.py

# Ensure you're in the correct directory
pwd
ls -la chatbot.py
```

#### 5. "Input not available" Message

**Problem**: Running in an environment without interactive input.

**Solution**: This is normal behavior in non-interactive environments. The chatbot will run a demo conversation automatically.

### Debug Mode

For detailed debugging, you can enable LangSmith tracing:

```bash
# Add to your .env file
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your-langsmith-api-key-here
```

### Getting Help

If you encounter issues not covered here:

1. Check the error message carefully - it usually indicates the specific problem
2. Verify your API keys are valid and have sufficient quota
3. Ensure all dependencies are installed correctly
4. Try with a different LLM provider
5. Check the [LangGraph documentation](https://langchain-ai.github.io/langgraph/) for updates

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)

## 📄 License

This project is provided as-is for educational and development purposes. Please ensure you comply with the terms of service for any LLM providers you use.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this chatbot implementation.

---

**Happy Chatting! 🚀**
