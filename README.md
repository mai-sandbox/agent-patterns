# LangGraph Agent Interaction & Feedback App

A Streamlit application that connects to a LangGraph agent deployed using `langgraph dev`, providing real-time streaming interactions and a feedback loop system to improve agent responses through iterative configuration updates.

## 🏗️ Architecture

This application consists of:

- **LangGraph Agent** (`src/agent.py`): A simple chat agent with tool calling capabilities using StateGraph, ToolNode, and tools_condition
- **Streamlit App** (`app.py`): Web interface for interacting with the agent, collecting feedback, and updating configurations
- **Configuration** (`langgraph.json`): LangGraph server configuration with configurable system prompt
- **Startup Script** (`run.sh`): Simple script to launch the LangGraph development server

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root or set environment variables:

```bash
# Required: OpenAI API Key for the language model
export OPENAI_API_KEY="your-openai-api-key-here"

# Required: Tavily API Key for web search functionality
export TAVILY_API_KEY="your-tavily-api-key-here"
```

#### Getting API Keys:

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/) for web search capabilities

### 4. Launch the LangGraph Server

Use the provided startup script:

```bash
./run.sh
```

Or manually:

```bash
langgraph dev
```

The server will be available at:
- **API**: http://localhost:2024
- **LangGraph Studio**: http://localhost:2024/studio

### 5. Launch the Streamlit App

In a new terminal window:

```bash
streamlit run app.py
```

The Streamlit app will be available at: http://localhost:8501

## 📋 Detailed Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API account
- Tavily API account (for web search)

### Environment Variables Setup

#### Option 1: Using .env file (Recommended)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

#### Option 2: Export in terminal

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export TAVILY_API_KEY="your-tavily-api-key-here"
```

#### Option 3: Set in your shell profile

Add to your `~/.bashrc`, `~/.zshrc`, or equivalent:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export TAVILY_API_KEY="your-tavily-api-key-here"
```

### Installing Dependencies

The `requirements.txt` includes:

- `langgraph>=0.2.0` - Core LangGraph framework
- `langchain-openai>=0.2.0` - OpenAI integration
- `langchain-tavily>=0.2.0` - Tavily search tool
- `streamlit>=1.28.0` - Web app framework
- `langgraph-sdk>=0.1.0` - Client SDK for connecting to LangGraph server

Install with:

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Start the LangGraph Server

The LangGraph server hosts your agent and provides the API:

```bash
./run.sh
```

Wait for the server to start (you'll see "Server ready" message).

### 2. Launch the Streamlit App

In a new terminal:

```bash
streamlit run app.py
```

### 3. Interact with the Agent

1. **Chat**: Type messages in the chat input
2. **Configure**: Modify the system prompt in the sidebar
3. **Feedback Loop**: 
   - After each agent response, provide feedback
   - Submit feedback to update the system prompt
   - Use "Rerun with Updated Config" to test improvements

## 🔧 Features

### Agent Capabilities

- **Chat Interface**: Natural language conversation
- **Tool Calling**: Web search using Tavily
- **Configurable System Prompt**: Customize agent behavior
- **Real-time Streaming**: See responses as they're generated

### Feedback System

- **Post-Response Feedback**: Collect user feedback after each interaction
- **Dynamic Configuration**: Update system prompt based on feedback
- **Iterative Improvement**: Rerun queries with updated configuration
- **Persistent Learning**: Feedback is incorporated into future responses

## 🛠️ Development

### Project Structure

```
.
├── src/
│   ├── __init__.py
│   └── agent.py          # LangGraph agent implementation
├── app.py                # Streamlit application
├── langgraph.json        # LangGraph server configuration
├── requirements.txt      # Python dependencies
├── run.sh               # Startup script
└── README.md            # This file
```

### Customizing the Agent

Edit `src/agent.py` to:
- Add new tools
- Modify the agent logic
- Change the state schema
- Update the graph structure

### Customizing the UI

Edit `app.py` to:
- Modify the Streamlit interface
- Add new feedback mechanisms
- Change the streaming behavior
- Update the configuration options

## 🐛 Troubleshooting

### Common Issues

1. **"LangGraph CLI not found"**
   - Ensure you've installed dependencies: `pip install -r requirements.txt`

2. **"Failed to connect to LangGraph server"**
   - Make sure the LangGraph server is running: `./run.sh`
   - Check that port 2024 is not in use by another application

3. **"API Key not set" warnings**
   - Set your environment variables as described above
   - Restart your terminal after setting environment variables

4. **Import errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility (3.8+)

### Getting Help

- Check the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- Review the [Streamlit Documentation](https://docs.streamlit.io/)
- Open an issue in this repository

## 📄 License

This project is open source and available under the MIT License.

