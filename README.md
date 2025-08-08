# Support Ticket Triage Agent

An advanced LangGraph-based support ticket triage system that automatically classifies, prioritizes, summarizes, routes, and drafts acknowledgements for customer support tickets.

## Features

- **Intelligent Classification**: Automatically categorizes tickets into Billing, Technical, or General Inquiry
- **Priority Detection**: Analyzes urgency indicators to assign Low, Medium, or High priority levels
- **Smart Summarization**: Creates concise one-sentence summaries of ticket content
- **Conditional Routing**: Routes tickets to appropriate email addresses based on category and priority
- **Acknowledgement Generation**: Drafts professional acknowledgement email snippets
- **Robust Error Handling**: Graceful fallbacks for all processing steps
- **Type Safety**: Comprehensive type hints throughout the codebase

## Architecture

The agent uses LangGraph's StateGraph to implement a 5-step sequential workflow:

```
Ticket Input → Classify → Prioritize → Summarize → Route → Acknowledge → Result
```

### Routing Logic

| Category | Priority | Email Address |
|----------|----------|---------------|
| Billing | High | priority-billing@company.com |
| Billing | Medium/Low | billing@company.com |
| Technical | High | urgent-tech@company.com |
| Technical | Medium/Low | tech@company.com |
| General Inquiry | Any | support@company.com |

## Installation

### Prerequisites

- Python 3.8 or higher
- Anthropic API key

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   # Ensure you have these files in your project directory:
   # - agent.py
   # - state.py
   # - requirements.txt
   # - .env.example
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   
   # On Unix/macOS:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your Anthropic API key
   # ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

5. **Get your Anthropic API key**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create an account or sign in
   - Generate an API key
   - Add it to your `.env` file

## Usage

### Basic Usage

```python
from agent import SupportTicketTriageAgent

# Initialize the agent
agent = SupportTicketTriageAgent()

# Process a support ticket
ticket_text = """
Hi, I'm having trouble logging into my account. I've tried resetting my password 
multiple times but I keep getting an error message saying 'Invalid credentials'. 
This is really urgent because I need to access my project files for a client 
presentation tomorrow morning. Can someone please help me ASAP?
"""

# Process the ticket through the complete workflow
result = agent.process_ticket(ticket_text)

# Access the results
print(f"Category: {result['category']}")           # Technical
print(f"Priority: {result['priority']}")           # High
print(f"Summary: {result['summary']}")             # Customer experiencing login issues...
print(f"Routed to: {result['email']}")             # urgent-tech@company.com
print(f"Acknowledgement: {result['acknowledgement']}")  # Thank you for contacting us...
```

### Command Line Usage

Run the included example:

```bash
python agent.py
```

This will process a sample ticket and display the complete triage results.

### Custom Model Configuration

```python
# Use a different Claude model
agent = SupportTicketTriageAgent(model_name="claude-3-sonnet-20240229")

# The agent will automatically use environment variables for configuration
```

## Example Results

### Sample Billing Ticket (High Priority)

**Input:**
```
URGENT: My credit card was charged twice for the same subscription! 
I need this fixed immediately as it's causing overdraft fees. 
This is completely unacceptable!
```

**Output:**
```
Category: Billing
Priority: High
Summary: Customer was charged twice for subscription causing overdraft fees.
Routed to: priority-billing@company.com
Acknowledgement: Thank you for contacting us about the duplicate billing charge. Your urgent billing issue has been escalated to our priority billing team for immediate resolution.
```

### Sample Technical Ticket (Medium Priority)

**Input:**
```
I'm having some issues with the search feature in your app. 
When I try to search for documents, sometimes it doesn't return 
all the results I expect. It's not urgent but would be nice to fix.
```

**Output:**
```
Category: Technical
Priority: Medium
Summary: Customer experiencing inconsistent search results in the application.
Routed to: tech@company.com
Acknowledgement: Thank you for reporting the search functionality issue. Your technical support request has been forwarded to our development team for investigation.
```

### Sample General Inquiry (Low Priority)

**Input:**
```
Hi, I was wondering if you have any plans to add dark mode to your application? 
I really like using dark themes and it would be a great addition.
```

**Output:**
```
Category: General Inquiry
Priority: Low
Summary: Customer inquiring about potential dark mode feature addition.
Routed to: support@company.com
Acknowledgement: Thank you for your feature suggestion regarding dark mode. Your inquiry has been forwarded to our support team who will provide information about our product roadmap.
```

## Environment Configuration

The agent uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

### Required Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)

### Optional Variables

- `ANTHROPIC_MODEL`: Claude model to use (default: claude-3-haiku-20240307)
- `LLM_TEMPERATURE`: Model temperature (default: 0.1)
- `LLM_MAX_TOKENS`: Maximum tokens per response (default: 1000)
- `LOG_LEVEL`: Logging level (default: INFO)

## Error Handling

The agent includes comprehensive error handling:

- **Invalid API Key**: Clear error message with setup instructions
- **Empty Ticket Text**: Validation with helpful error message
- **LLM Failures**: Graceful fallbacks to default values
- **Network Issues**: Automatic retry logic (built into LangChain)

### Common Error Scenarios

1. **Missing API Key**
   ```
   ValueError: ANTHROPIC_API_KEY environment variable is required
   ```
   **Solution**: Add your API key to the `.env` file

2. **Empty Ticket**
   ```
   ValueError: Ticket text cannot be empty
   ```
   **Solution**: Provide non-empty ticket text

3. **Classification Failures**
   - Defaults to "General Inquiry" category
   - Defaults to "Medium" priority
   - Creates basic summary and acknowledgement

## Troubleshooting

### Installation Issues

**Problem**: `pip install` fails with dependency conflicts
```bash
# Solution: Use a fresh virtual environment
python -m venv fresh_venv
source fresh_venv/bin/activate  # or fresh_venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Problem**: Import errors for LangGraph or LangChain
```bash
# Solution: Ensure you're in the correct virtual environment
which python  # Should point to your venv
pip list | grep langgraph  # Should show installed version
```

### Runtime Issues

**Problem**: Slow processing times
- **Cause**: Using a larger Claude model
- **Solution**: Use `claude-3-haiku-20240307` for faster responses

**Problem**: Inconsistent classifications
- **Cause**: Model temperature too high
- **Solution**: Set `LLM_TEMPERATURE=0.1` in `.env` file

**Problem**: API rate limits
- **Cause**: Too many requests to Anthropic API
- **Solution**: Implement request throttling or upgrade API plan

### Debugging

Enable debug logging by setting environment variables:

```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
python agent.py
```

## Development

### Code Quality

The project includes code quality tools:

```bash
# Run linting
ruff check .

# Run type checking
mypy . --ignore-missing-imports

# Fix auto-fixable issues
ruff check . --fix
```

### Project Structure

```
support-ticket-triage/
├── agent.py              # Main LangGraph implementation
├── state.py              # State schema definitions
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── .env                  # Your environment variables (not in git)
└── README.md            # This documentation
```

### Extending the Agent

To add new functionality:

1. **Add new state fields** in `state.py`
2. **Create new node functions** in `agent.py`
3. **Update the workflow graph** in `_build_graph()`
4. **Add comprehensive error handling**
5. **Update type hints and documentation**

## API Reference

### SupportTicketTriageAgent

#### `__init__(model_name: str = "claude-3-haiku-20240307")`
Initialize the agent with specified Claude model.

#### `process_ticket(ticket_text: str) -> TicketState`
Process a support ticket through the complete triage workflow.

**Parameters:**
- `ticket_text`: Raw customer support ticket text

**Returns:**
- `TicketState`: Complete state with all fields populated

**Raises:**
- `ValueError`: If ticket_text is empty or API key is missing

### TicketState

TypedDict containing:
- `ticket_text: str`: Original ticket content
- `category: str`: Classification result
- `priority: str`: Priority level
- `summary: str`: One-sentence summary
- `email: str`: Routed email address
- `acknowledgement: str`: Draft acknowledgement

## License

This project is provided as-is for educational and development purposes.

## Support

For issues with this implementation:
1. Check the troubleshooting section above
2. Verify your environment setup
3. Review the error messages for specific guidance
4. Ensure your Anthropic API key is valid and has sufficient credits
