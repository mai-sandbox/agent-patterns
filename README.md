# LangGraph Support Ticket Triage Agent

An advanced support ticket triage system built with LangGraph that automatically processes customer support tickets through a structured 5-step workflow. The agent classifies tickets, detects priority levels, generates summaries, routes to appropriate teams, and drafts acknowledgment responses.

## Features

- **Intelligent Classification**: Automatically categorizes tickets into Billing, Technical, or General Inquiry
- **Priority Detection**: Analyzes urgency clues to assign Low, Medium, or High priority levels
- **Smart Summarization**: Generates concise one-sentence summaries of customer issues
- **Automated Routing**: Routes tickets to appropriate email addresses based on category and priority
- **Response Generation**: Creates professional acknowledgment email snippets
- **Robust Error Handling**: Comprehensive error handling with fallback mechanisms
- **Structured Logging**: Detailed logging for monitoring and debugging

## Workflow Overview

The agent processes tickets through these sequential steps:

1. **Classification Node** → Categorizes ticket (Billing/Technical/General Inquiry)
2. **Priority Detection Node** → Determines urgency level (Low/Medium/High)
3. **Summarization Node** → Creates one-sentence summary
4. **Routing Node** → Assigns email based on category + priority rules
5. **Acknowledgment Node** → Drafts 1-2 sentence response snippet

### Email Routing Rules

| Category | Priority | Routed To |
|----------|----------|-----------|
| Billing | High | priority-billing@company.com |
| Billing | Medium/Low | billing@company.com |
| Technical | High | urgent-tech@company.com |
| Technical | Medium/Low | tech@company.com |
| General Inquiry | Any | support@company.com |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

If you have this code in a repository:
```bash
git clone <repository-url>
cd agent-patterns
```

Or download the files directly to your project directory.

### Step 2: Create Virtual Environment

Create and activate a Python virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

This will install:
- LangGraph and LangChain core libraries
- Anthropic Claude integration
- Environment variable management
- Code quality tools (ruff, mypy)
- Additional supporting libraries

### Step 4: Environment Configuration

Set up your environment variables for LLM API access:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
# Use your preferred text editor
nano .env
# or
code .env
```

**Required Configuration:**
- Get your Anthropic API key from [https://console.anthropic.com/](https://console.anthropic.com/)
- Replace `your_anthropic_api_key_here` with your actual API key in the `.env` file

## Usage

### Basic Usage

Here's how to use the support ticket triage agent:

```python
from support_triage_agent import SupportTriageAgent

# Initialize the agent
agent = SupportTriageAgent()

# Example support ticket
ticket_text = """
Hi there,

I'm having trouble with my billing. I was charged twice for my subscription 
this month and need this resolved immediately as it's affecting my budget.

Please help!
Thanks,
Sarah
"""

# Process the ticket
result = agent.process_ticket(ticket_text)

# View the results
print(f"Category: {result['category']}")
print(f"Priority: {result['priority']}")
print(f"Summary: {result['summary']}")
print(f"Routing Email: {result['routing_email']}")
print(f"Acknowledgment: {result['acknowledgment']}")
```

### Expected Output

```
Category: Billing
Priority: High
Summary: Customer was charged twice for subscription and needs immediate resolution.
Routing Email: priority-billing@company.com
Acknowledgment: Thank you for contacting us about the duplicate billing charge. Your ticket has been forwarded to our priority billing team for immediate resolution.
```

### Running the Example

The agent includes a built-in example. Run it directly:

```bash
python support_triage_agent.py
```

This will process a sample technical support ticket and display the complete triage results.

### Advanced Usage

#### Custom Configuration

```python
# Initialize with custom API key
agent = SupportTriageAgent(api_key="your-custom-api-key")

# Process multiple tickets
tickets = [
    "I can't log into my account...",
    "When will the new feature be available?",
    "My payment failed, please help!"
]

for i, ticket in enumerate(tickets):
    print(f"\n--- Processing Ticket {i+1} ---")
    result = agent.process_ticket(ticket)
    print(f"Routed to: {result['routing_email']}")
```

#### Accessing Routing Rules

```python
# View current routing rules
rules = agent.get_routing_rules()
for (category, priority), email in rules.items():
    print(f"{category} + {priority} → {email}")
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic Claude API key | `sk-ant-api03-...` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key for debugging | - |
| `LANGCHAIN_PROJECT` | LangSmith project name | - |
| `LOG_LEVEL` | Logging level | `INFO` |

### Getting API Keys

1. **Anthropic API Key** (Required):
   - Visit [https://console.anthropic.com/](https://console.anthropic.com/)
   - Sign up or log in to your account
   - Navigate to the API Keys section
   - Create a new API key
   - Copy and paste it into your `.env` file

2. **LangSmith API Key** (Optional - for debugging):
   - Visit [https://smith.langchain.com/](https://smith.langchain.com/)
   - Create an account
   - Generate an API key
   - Add it to your `.env` file for enhanced debugging

## Project Structure

```
agent-patterns/
├── support_triage_agent.py    # Main agent implementation
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .env                      # Your actual environment variables (create this)
├── README.md                 # This documentation
└── .gitignore               # Git ignore rules
```

## Troubleshooting

### Common Issues

#### 1. "Anthropic API key is required" Error

**Problem**: The agent can't find your API key.

**Solutions**:
- Ensure you've copied `.env.example` to `.env`
- Verify your API key is correctly set in the `.env` file
- Check for extra spaces or quotes around the API key
- Make sure the `.env` file is in the same directory as the script

#### 2. Import Errors

**Problem**: Python can't find required packages.

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep langgraph
```

#### 3. API Rate Limiting

**Problem**: Getting rate limit errors from Anthropic.

**Solutions**:
- Check your Anthropic account usage limits
- Consider upgrading your API plan
- Implement retry logic for production use
- Add delays between requests if processing many tickets

#### 4. Permission Errors

**Problem**: Can't read `.env` file or write logs.

**Solutions**:
```bash
# Check file permissions
ls -la .env

# Fix permissions if needed
chmod 644 .env

# Ensure directory is writable
chmod 755 .
```

#### 5. Unexpected Classification Results

**Problem**: Agent is misclassifying tickets.

**Solutions**:
- Review the ticket text for clarity
- Check if the issue spans multiple categories
- Consider the LLM's interpretation of ambiguous requests
- Add more specific language to your tickets for better results

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your agent
agent = SupportTriageAgent()
result = agent.process_ticket("Your ticket text here")
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: The agent provides detailed logging information
2. **Verify your setup**: Ensure all installation steps were completed
3. **Test with simple examples**: Start with basic ticket text
4. **Check API status**: Verify Anthropic's service status
5. **Review environment**: Ensure your Python environment is properly configured

## Development

### Code Quality

This project uses code quality tools to maintain high standards:

```bash
# Run linting
ruff check .

# Run type checking
mypy . --ignore-missing-imports

# Format code
ruff format .
```

### Testing

Test the agent with various ticket types:

```python
# Test different categories
test_tickets = {
    "billing": "I need a refund for my last payment",
    "technical": "The app crashes when I try to login",
    "general": "What are your business hours?"
}

for category, ticket in test_tickets.items():
    result = agent.process_ticket(ticket)
    print(f"Expected: {category}, Got: {result['category'].lower()}")
```

## Security Considerations

- **Never commit your `.env` file** to version control
- **Rotate API keys regularly** for security
- **Use environment-specific keys** for development/staging/production
- **Monitor API usage** to detect unusual activity
- **Implement rate limiting** in production environments

## License

This project is provided as-is for educational and development purposes. Please ensure you comply with the terms of service for any APIs you use (Anthropic, LangSmith, etc.).

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Add type hints to all functions
3. Include comprehensive error handling
4. Update documentation for any changes
5. Test with various ticket types
6. Run code quality checks before submitting

---

**Need help?** Check the troubleshooting section above or review the example usage patterns provided in the code.
