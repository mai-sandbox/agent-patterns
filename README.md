# Multi-Functional Web-Enabled Agent

A powerful LangGraph-based AI agent that provides weather information, web search capabilities, and insights into current ice cream flavor trends. Built with LangChain and powered by Claude AI.

## Features

🌤️ **Weather Information**
- Get current weather conditions for any location worldwide
- Detailed information including temperature, humidity, wind speed, and more
- Powered by OpenWeatherMap API

🔍 **Web Search**
- Search the web for real-time information
- Get relevant results with titles, URLs, and content snippets
- Powered by Tavily Search Engine

🍦 **Ice Cream Flavor Trends**
- Discover the world's current favorite ice cream flavors
- Get insights from recent surveys and trend data
- Stay updated with the latest flavor preferences

## Prerequisites

- Python 3.8 or higher
- API keys for the following services:
  - [Anthropic](https://console.anthropic.com/) (for Claude AI)
  - [Tavily](https://tavily.com/) (for web search)
  - [OpenWeatherMap](https://openweathermap.org/api) (for weather data)

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd agent-patterns
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## API Key Configuration

1. **Copy the environment template**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file and add your API keys**
   ```env
   # Anthropic API key for Claude LLM
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
   
   # Tavily API key for web search functionality
   TAVILY_API_KEY=your_actual_tavily_api_key_here
   
   # OpenWeatherMap API key for weather information
   OPENWEATHERMAP_API_KEY=your_actual_openweathermap_api_key_here
   ```

### Getting API Keys

**Anthropic API Key:**
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key

**Tavily API Key:**
1. Visit [Tavily](https://tavily.com/)
2. Sign up for an account
3. Get your API key from the dashboard
4. Note: Free tier includes 1000 searches per month

**OpenWeatherMap API Key:**
1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Go to API Keys section
4. Generate a new API key

## Usage

1. **Start the agent**
   ```bash
   python main.py
   ```

2. **Interact with the agent**
   The agent will start an interactive conversation. You can ask questions like:

   **Weather queries:**
   - "What's the weather like in London?"
   - "Tell me about the current weather in Tokyo, Japan"
   - "How's the weather in New York, US?"

   **Web search queries:**
   - "Search for the latest news about artificial intelligence"
   - "Find information about Python programming"
   - "What are the current trends in technology?"

   **Ice cream flavor queries:**
   - "What's the world's current favorite ice cream flavor?"
   - "Tell me about popular ice cream flavors"
   - "What are the trending ice cream flavors?"

3. **Exit the conversation**
   Type any of the following to exit:
   - `quit`
   - `exit`
   - `bye`
   - `q`
   - Or press `Ctrl+C`

## Project Structure

```
agent-patterns/
├── main.py              # Main application entry point
├── tools.py             # Custom tool implementations
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .env                 # Your actual API keys (create this)
├── README.md           # This file
└── Agent.md            # Development guidelines
```

## Development
