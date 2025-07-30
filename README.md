# Form-Filling Agent

A sophisticated LangGraph-based agent that can fill out forms section by section with human-in-the-loop validation, persistent memory, and comprehensive error handling.

## 🌟 Features

- **Section-by-Section Processing**: Processes forms systematically, one section at a time
- **Human-in-the-Loop Validation**: Pauses for human review and validation at each section
- **Persistent Memory**: Maintains state across sessions using LangGraph checkpointing
- **Comprehensive Error Handling**: Robust error handling with retry logic and recovery mechanisms
- **Progress Tracking**: Real-time progress monitoring with completion percentages
- **Type-Safe State Management**: Full type hints and validation using TypedDict
- **Flexible Form Definitions**: Easily configurable form sections and field definitions
- **Production Ready**: Deployment-ready with proper configuration and error handling

## 📋 Form Sections

The agent processes forms with the following default sections:

1. **Personal Information**: Name, date of birth, gender
2. **Contact Details**: Email, phone, address, city, postal code
3. **Employment Information**: Company, position, experience, salary range
4. **Preferences**: Communication method, newsletter subscription, special requirements

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com/))

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd agent-patterns
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

### Basic Usage

#### Option 1: Simple Example
```bash
python example.py
```

#### Option 2: Interactive Demo (Async)
```bash
python example.py --async
```

#### Option 3: Direct Agent Usage
```python
from agent import app
from state import create_initial_state, DEFAULT_FORM_SECTIONS

# Create configuration with thread ID for persistence
config = {"configurable": {"thread_id": "my_form_session"}}

# Start the form-filling process
initial_input = {
    "messages": [{"role": "user", "content": "I'd like to fill out a form."}]
}

# Run the agent
result = app.invoke(initial_input, config)
print(result["messages"][-1].content)
```

## 📖 Detailed Usage Guide

### Starting a New Form

The agent automatically initializes with predefined form sections. When you start a conversation, it will:

1. Welcome you and explain the process
2. Show you all sections that need to be completed
3. Begin with the first section (Personal Information)

### Processing Sections Step-by-Step

For each section, the agent will:

1. **Explain the section**: Describe what information is needed
2. **Collect information**: Ask for required and optional fields conversationally
3. **Validate data**: Use the validation tool to check completeness and accuracy
4. **Request human review**: Pause for your confirmation or corrections
5. **Mark complete**: Move to the next section once validated

### Human-in-the-Loop Validation

When the agent requests validation, you'll see:

- **Section name** and **data collected**
- **Validation errors** (if any)
- **Options for response**:
  - `yes`/`y` - Approve the data as-is
  - `no`/`n` - Make corrections
  - `skip` - Skip this section
  - `restart` - Restart the section from scratch

### Example Validation Flow

```
🤖 HUMAN VALIDATION REQUIRED 🤖
==================================================

Section: Personal Information
Question: Please review the personal information section data. Is this correct?

Data to Review:
  • First Name: John
  • Last Name: Doe
  • Date Of Birth: 1990-01-15
  • Gender: Male

Options:
  1. Type 'yes' or 'y' to approve the data
  2. Type 'no' or 'n' to make corrections
  3. Type 'skip' to skip this section
  4. Type 'restart' to restart this section

👤 Your choice: yes
```

### Handling Corrections

If you choose to make corrections:

1. The agent will go through each field
2. You can provide new values or press Enter to keep current values
3. The agent will retry validation with your corrections
4. Maximum of 3 retry attempts per section

### Form Completion

Once all sections are completed, the agent provides:

- **Comprehensive summary** of all collected data
- **Completion statistics** (sections, fields, percentages)
- **Form ID** for reference
- **Next steps** information

## 🏗️ Architecture

### Core Components

- **`agent.py`**: Main LangGraph StateGraph implementation
- **`state.py`**: Custom state schema and utility functions
- **`example.py`**: CLI interface and usage examples
- **`requirements.txt`**: Python dependencies
- **`langgraph.json`**: LangGraph server configuration
- **`.env.example`**: Environment variable template

### State Management

The agent uses a comprehensive state schema:

```python
class FormFillingState(TypedDict):
    messages: Annotated[List[Any], add_messages]  # Conversation history
    form_data: Dict[str, Dict[str, Any]]          # Collected form data
    current_section: str                          # Current section being processed
    sections_completed: List[str]                 # Completed sections
    total_sections: int                           # Total number of sections
    current_field: Optional[str]                  # Current field (optional)
    validation_errors: Optional[List[str]]        # Validation errors (optional)
    user_preferences: Optional[Dict[str, Any]]    # User preferences (optional)
```

### Workflow Nodes

1. **`start_form`**: Initialize the form-filling process
2. **`process_section`**: Handle section processing with LLM assistance
3. **`tools`**: Execute validation tools
4. **`validation`**: Process validation results
5. **`mark_section_complete`**: Mark sections as complete and advance
6. **`completion`**: Handle form completion and generate summary

### Error Handling

The agent includes comprehensive error handling:

- **Validation errors**: Field-level validation with specific error messages
- **Retry logic**: Up to 3 attempts per section with correction support
- **Critical errors**: System-level errors with recovery mechanisms
- **Form integrity**: State consistency validation
- **LLM errors**: Graceful handling of model failures

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required: Anthropic API key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Alternative LLM providers
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: LangSmith tracing (for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=form-filling-agent

# Optional: Application configuration
LOG_LEVEL=INFO
```

### Custom Form Definitions

You can customize form sections by modifying the `FORM_FIELDS` dictionary in `agent.py`:

```python
FORM_FIELDS = {
    "your_section": {
        "field_name": {
            "type": "str",           # str, int, bool
            "required": True,        # True or False
            "description": "Field description for user"
        }
    }
}
```

### LangGraph Server Configuration

The `langgraph.json` file configures the agent for local development:

```json
{
  "dependencies": ["."],
  "graphs": {
    "form_filling_agent": "./agent.py:app"
  },
  "env": ".env"
}
```

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements.txt

# Run the example script
python example.py

# Test with different thread IDs for multiple sessions
python -c "
from agent import app
config1 = {'configurable': {'thread_id': 'test1'}}
config2 = {'configurable': {'thread_id': 'test2'}}
# Each config maintains separate state
"
```

### Code Quality

The project follows strict code quality standards:

```bash
# Check code formatting and style
ruff check .

# Check type hints
mypy . --ignore-missing-imports

# Format code (if using ruff format)
ruff format .
```

### Local Development Server

You can run the agent with the LangGraph development server:

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Start the development server
langgraph dev

# Access the agent at http://localhost:8123
```

## 📊 Monitoring and Debugging

### LangSmith Integration

Enable LangSmith tracing for detailed execution monitoring:

1. Set up LangSmith account at [smith.langchain.com](https://smith.langchain.com)
2. Add your API key to `.env`
3. Set `LANGCHAIN_TRACING_V2=true`
4. View traces in the LangSmith dashboard

### State Inspection

You can inspect the agent's state at any time:

```python
from agent import app

config = {"configurable": {"thread_id": "your_session"}}
state = app.get_state(config)

print("Current state:", state.values)
print("Next nodes:", state.next)
print("Completed sections:", state.values.get("sections_completed", []))
```

### Progress Tracking

Monitor form completion progress:

```python
from state import get_completion_percentage, get_form_progress_summary

# Get completion percentage
progress = get_completion_percentage(state.values)
print(f"Form {progress:.1f}% complete")

# Get detailed progress summary
summary = get_form_progress_summary(state.values)
print(summary)
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: ANTHROPIC_API_KEY not found in environment variables
   ```
   **Solution**: Ensure your `.env` file contains a valid `ANTHROPIC_API_KEY`

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langgraph'
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

3. **State Persistence Issues**
   ```
   Warning: State not persisting between sessions
   ```
   **Solution**: Ensure you're using the same `thread_id` in your config

4. **Validation Errors**
   ```
   Validation failed: Missing required field
   ```
   **Solution**: Provide all required fields or use the correction flow

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your agent code
```

### Getting Help

- Check the [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- Review the `example.py` script for usage patterns
- Inspect the agent state using `app.get_state(config)`
- Enable LangSmith tracing for detailed execution logs

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Add type hints to all functions
3. Include comprehensive error handling
4. Test your changes with the example script
5. Run code quality checks before submitting

---

**Happy form filling! 🎉**
