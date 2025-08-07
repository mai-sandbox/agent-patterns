# Basic Chatbot - LangGraph Implementation

A conversational chatbot built with LangGraph and Anthropic Claude, featuring proper state management, comprehensive error handling, and production-ready logging capabilities.

## Features

- 🤖 **LangGraph StateGraph**: Modern conversation flow management
- 🧠 **Anthropic Claude Integration**: Powered by Claude-3-Haiku for intelligent responses
- 💬 **Conversation History**: Maintains context across multiple exchanges
- 🔧 **Configurable Settings**: Customizable model parameters and logging
- 🛡️ **Error Handling**: Comprehensive error management with graceful fallbacks
- 📝 **Logging**: Detailed logging with configurable levels
- 🎯 **Type Safety**: Full type hints for better development experience

## Requirements

- Python 3.8 or higher
- Anthropic API key

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agent-patterns
```

### 2. Create Virtual Environment

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

### 4. Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your Anthropic API key:
   ```bash
   # Required: Get your API key from https://console.anthropic.com/
   ANTHROPIC_API_KEY=your_actual_api_key_here
   
   # Optional: Customize other settings
   ANTHROPIC_MODEL=claude-3-haiku-20240307
   MAX_TOKENS=1000
   TEMPERATURE=0.7
   LOG_LEVEL=INFO
   DEBUG_MODE=false
   ```

## Usage

### Interactive Command Line

Run the chatbot in interactive mode:

```bash
python chatbot.py
```

This will start an interactive session where you can:
- Chat with the bot by typing messages
- Use `help` to see available commands
- Use `clear` to reset conversation history
- Use `quit`, `exit`, or `bye` to end the session

### Programmatic Usage

You can also use the chatbot programmatically in your own Python code:

```python
from chatbot import BasicChatbot

# Initialize the chatbot
bot = BasicChatbot()

# Create a session
session_id = bot.create_session(user_id="user123")

# Single message chat
response = bot.chat("Hello, how are you?", session_id)
print(response)

# Chat with conversation history
from langchain_core.messages import HumanMessage, AIMessage

conversation_history = []
response, updated_history = bot.chat_with_history(
    "What's the weather like?", 
    conversation_history, 
    session_id
)
print(response)

# Continue the conversation
response, updated_history = bot.chat_with_history(
    "What about tomorrow?", 
    updated_history, 
    session_id
)
print(response)
```

### Configuration Options

The chatbot can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) | - |
| `ANTHROPIC_MODEL` | Claude model to use | `claude-3-haiku-20240307` |
| `MAX_TOKENS` | Maximum tokens in response | `1000` |
| `TEMPERATURE` | Response creativity (0.0-1.0) | `0.7` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEBUG_MODE` | Enable debug logging to file | `false` |

## Project Structure

```
agent-patterns/
├── chatbot.py          # Main chatbot implementation
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variable template
├── .env              # Your environment variables (create this)
├── README.md         # This documentation
└── chatbot.log       # Log file (created when DEBUG_MODE=true)
```

## API Reference

### BasicChatbot Class

#### `__init__()`
Initialize the chatbot with configuration from environment variables.

**Raises:**
- `ChatbotError`: If required environment variables are missing or invalid

#### `create_session(user_id: Optional[str] = None) -> str`
Create a new conversation session.

**Parameters:**
- `user_id`: Optional user identifier

**Returns:**
- `str`: Unique session identifier

#### `chat(message: str, session_id: str, user_id: Optional[str] = None) -> str`
Process a single chat message.

**Parameters:**
- `message`: User's input message
- `session_id`: Session identifier
- `user_id`: Optional user identifier

**Returns:**
- `str`: Bot's response

**Raises:**
- `ChatbotError`: If message processing fails

#### `chat_with_history(message: str, conversation_history: List[BaseMessage], session_id: str, user_id: Optional[str] = None) -> tuple[str, List[BaseMessage]]`
Process a chat message with conversation history.

**Parameters:**
- `message`: User's input message
- `conversation_history`: Previous conversation messages
- `session_id`: Session identifier
- `user_id`: Optional user identifier

**Returns:**
- `tuple`: (Bot's response, Updated conversation history)

**Raises:**
- `ChatbotError`: If message processing fails

### ChatState Schema

The conversation state includes:
- `messages`: List of conversation messages
- `user_id`: Optional user identifier
- `session_id`: Unique session identifier
- `created_at`: Timestamp when conversation started

## Development

### Code Quality

This project uses `ruff` for linting and `mypy` for type checking:

```bash
# Run linting
ruff check .

# Run type checking
mypy . --ignore-missing-imports
```

### Logging

The chatbot includes comprehensive logging:
- Console output for general information
- Optional file logging when `DEBUG_MODE=true`
- Configurable log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Troubleshooting

### Common Issues

#### "ANTHROPIC_API_KEY not found or not set"
**Problem:** The API key is missing or incorrectly configured.

**Solution:**
1. Ensure you've created a `.env` file from `.env.example`
2. Add your actual Anthropic API key to the `.env` file
3. Make sure the key is not set to the placeholder value `your_anthropic_api_key_here`

#### "Failed to initialize chatbot: Invalid API key"
**Problem:** The provided API key is invalid or expired.

**Solution:**
1. Verify your API key at https://console.anthropic.com/
2. Generate a new API key if necessary
3. Update the `.env` file with the correct key

#### "Chat processing failed: Connection error"
**Problem:** Network connectivity issues or API service unavailable.

**Solution:**
1. Check your internet connection
2. Verify Anthropic API service status
3. Try again after a few moments
4. Check if you've exceeded API rate limits

#### "No response generated"
**Problem:** The LLM failed to generate a response.

**Solution:**
1. Check your API key and quota
2. Verify the message content is appropriate
3. Try reducing `MAX_TOKENS` if the request is too large
4. Check the logs for more detailed error information

#### Import Errors
**Problem:** Missing dependencies or incorrect Python version.

**Solution:**
1. Ensure you're using Python 3.8 or higher: `python --version`
2. Activate your virtual environment: `source venv/bin/activate`
3. Reinstall dependencies: `pip install -r requirements.txt`
4. If issues persist, try creating a fresh virtual environment

### Debug Mode

Enable debug mode for detailed logging:

1. Set `DEBUG_MODE=true` in your `.env` file
2. Set `LOG_LEVEL=DEBUG` for maximum verbosity
3. Check the `chatbot.log` file for detailed information

### Getting Help

If you encounter issues not covered here:

1. Check the log files for detailed error messages
2. Verify all environment variables are correctly set
3. Ensure your Python environment meets the requirements
4. Review the Anthropic API documentation for API-specific issues

## License

This project is provided as-is for educational and development purposes.

## Contributing

When contributing to this project:

1. Follow the existing code style and patterns
2. Include comprehensive type hints
3. Add appropriate error handling and logging
4. Update documentation as needed
5. Ensure all code quality checks pass (`ruff` and `mypy`)
