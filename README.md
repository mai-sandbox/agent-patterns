# LangGraph Form Filling Agent

A section-by-section form filling agent built with LangGraph that guides users through completing a multi-section form with validation and review capabilities.

## Features

- **Section-by-section processing**: Guides users through personal info, contact details, employment info, and review sections
- **State management**: Tracks form progress and completed sections
- **Intelligent routing**: Automatically advances to the next section when current section is complete
- **Memory persistence**: Maintains conversation state across interactions
- **Review and validation**: Allows users to review all collected information before submission
- **Conversational interface**: Natural language interaction for form completion

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   # Ensure you have all project files in your directory:
   # - agent.py
   # - langgraph.json
   # - requirements.txt
   # - .env.example
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

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your API keys:
   ```env
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
   LANGCHAIN_TRACING_V2=true  # Optional: for debugging
   LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Optional
   ```

## Environment Setup

### Required Environment Variables

- **ANTHROPIC_API_KEY**: Your Anthropic API key for Claude LLM access
  - Get your API key from: https://console.anthropic.com/

### Optional Environment Variables

- **LANGCHAIN_TRACING_V2**: Set to `true` to enable LangSmith tracing for debugging
- **LANGCHAIN_API_KEY**: Your LangSmith API key for tracing and monitoring
- **LANGCHAIN_PROJECT**: Project name for LangSmith (defaults to "form-filling-agent")

## Usage

### Basic Usage

```python
from agent import app

# Create a configuration for the conversation thread
config = {"configurable": {"thread_id": "user_session_1"}}

# Start the form filling process
initial_message = {"messages": [{"role": "user", "content": "I'd like to fill out a form"}]}

# Run the agent
for event in app.stream(initial_message, config):
    if "messages" in event:
        print(event["messages"][-1].content)
```

### Interactive Example

```python
from agent import app

def run_form_agent():
    config = {"configurable": {"thread_id": "interactive_session"}}
    
    print("Welcome to the Form Filling Agent!")
    print("I'll help you complete a form section by section.\n")
    
    # Initialize with a greeting
    state = {"messages": [{"role": "user", "content": "Hello, I need to fill out a form"}]}
    
    while True:
        # Get agent response
        for event in app.stream(state, config):
            if "messages" in event:
                print("Agent:", event["messages"][-1].content)
                
                # Check if form is complete
                if event.get("is_complete", False):
                    print("\nForm completed successfully!")
                    return
        
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
            
        # Update state with user message
        state = {"messages": [{"role": "user", "content": user_input}]}

if __name__ == "__main__":
    run_form_agent()
```

### Form Sections

The agent processes the following sections in order:

1. **Personal Information**
   - Full name
   - Date of birth
   - Gender (optional)
   - Nationality

2. **Contact Details**
   - Email address
   - Phone number
   - Street address
   - City, State, ZIP code

3. **Employment Information**
   - Current job title
   - Company name
   - Years of experience
   - Annual salary (optional)

4. **Review and Submit**
   - Review all collected information
   - Confirm or make corrections
   - Submit the completed form

## Deployment

### Local Development

For local testing and development:

```bash
# Install LangGraph CLI (if not already installed)
pip install langgraph-cli

# Run the agent locally
langgraph dev
```

### LangGraph Platform Deployment

The agent is configured for deployment on the LangGraph Platform using the `langgraph.json` configuration file.

#### Prerequisites for Deployment

1. **LangGraph Platform Account**: Sign up at the LangGraph Platform
2. **API Keys**: Ensure your environment variables are properly configured
3. **Dependencies**: All dependencies are specified in `requirements.txt`

#### Deployment Steps

1. **Using LangGraph CLI**:
   ```bash
   # Deploy to LangGraph Platform
   langgraph deploy
   ```

2. **Using GitHub Integration**:
   - Push your code to a GitHub repository
   - Connect the repository to LangGraph Platform
   - Configure environment variables in the platform
   - Deploy directly from the platform interface

#### Configuration Details

The `langgraph.json` file configures:
- **Dependencies**: Specifies required packages including the local package
- **Graphs**: Defines the `form_agent` graph pointing to `./agent.py:app`
- **Environment**: Loads variables from `.env` file

### Production Considerations

- **API Rate Limits**: Monitor Anthropic API usage and implement rate limiting if needed
- **Error Handling**: The agent includes basic error handling, but consider adding more robust error management for production
- **Logging**: Enable LangSmith tracing for monitoring and debugging in production
- **Security**: Ensure API keys are properly secured and not exposed in logs

## Architecture

### State Management

The agent uses a `FormState` TypedDict with the following fields:
- `messages`: Conversation history with automatic message aggregation
- `form_data`: Dictionary storing collected form information
- `current_section`: Tracks the current form section being processed
- `completed_sections`: List of sections that have been completed
- `is_complete`: Boolean indicating if the entire form is complete

### Graph Structure

```
START → personal_info → contact_details → employment_info → review_and_submit → END
         ↑                ↑                    ↑                    ↑
         └────────────────┴────────────────────┴────────────────────┘
                        (conditional routing based on state)
```

### Node Functions

- **personal_info_node**: Collects personal information
- **contact_details_node**: Collects contact information
- **employment_info_node**: Collects employment details
- **review_and_submit_node**: Reviews and finalizes the form

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `ANTHROPIC_API_KEY` is set in your `.env` file
   - Verify the API key is valid and has sufficient credits

2. **Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Verify you're using Python 3.8 or higher

3. **Graph Compilation Errors**
   - Check that all node functions return proper state updates
   - Ensure conditional edge functions return valid route names

### Debug Mode

Enable detailed logging by setting environment variables:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
```

## Contributing

1. Follow PEP 8 style guidelines
2. Include comprehensive type hints
3. Run code quality checks before submitting:
   ```bash
   ruff check .
   mypy . --ignore-missing-imports
   ```
4. Test the agent compilation and basic functionality

## License

This project is provided as-is for educational and development purposes.

## Support

For issues related to:
- **LangGraph**: Check the [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- **LangChain**: Visit the [LangChain documentation](https://python.langchain.com/)
- **Anthropic API**: See the [Anthropic documentation](https://docs.anthropic.com/)
