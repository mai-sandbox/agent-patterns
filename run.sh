#!/bin/bash

# LangGraph Agent Startup Script
# This script launches the LangGraph development server

echo "🚀 Starting LangGraph Agent Development Server..."
echo "=================================================="

# Check if langgraph is installed
if ! command -v langgraph &> /dev/null; then
    echo "❌ LangGraph CLI not found. Please install dependencies first:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. Please set your OpenAI API key."
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "⚠️  Warning: TAVILY_API_KEY not set. Please set your Tavily API key for web search."
fi

echo "🔧 Launching LangGraph development server..."
echo "📍 Server will be available at: http://localhost:2024"
echo "🎯 LangGraph Studio will be available at: http://localhost:2024/studio"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""

# Launch LangGraph development server
langgraph dev

