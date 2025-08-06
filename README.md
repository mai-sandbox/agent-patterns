# LangGraph Form Filling Agent

A sophisticated form filling agent built with LangGraph that processes forms section by section, providing an interactive and intelligent user experience. The agent uses AI to understand user input, extract relevant information, validate data, and guide users through complex forms step by step.

## 🌟 Project Overview

This LangGraph-powered agent transforms the traditional form filling experience by:

- **Sequential Processing**: Guides users through forms section by section for better completion rates
- **Intelligent Data Extraction**: Uses Claude AI to understand and structure user input automatically
- **Interactive Validation**: Provides real-time feedback and validation with helpful error messages
- **Human-in-the-Loop**: Allows users to review and correct information at each step
- **Flexible Architecture**: Easily configurable form sections and validation rules
- **Production Ready**: Deployable on LangGraph Platform with comprehensive monitoring

### Key Features

- 🤖 **AI-Powered**: Leverages Anthropic's Claude for intelligent form processing
- 📝 **Section-by-Section**: Breaks complex forms into manageable sections
- ✅ **Smart Validation**: Automatic validation with field-specific rules
- 🔄 **Error Recovery**: Graceful handling of validation errors with retry mechanisms
- 📊 **Progress Tracking**: Clear progress indicators and section completion status
- 🚀 **Deployment Ready**: Configured for LangGraph Platform deployment

## 📋 Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** (tested with Python 3.13.3)
- **Anthropic API Key** - Get yours at [console.anthropic.com](https://console.anthropic.com/settings/keys)
- **Git** - For cloning the repository
- **Virtual Environment** (recommended) - For dependency isolation

### Optional but Recommended

- **LangSmith Account** - For monitoring and debugging ([smith.langchain.com](https://smith.langchain.com/))
- **LangGraph Platform Account** - For cloud deployment

## 🚀 Installation Steps

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
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit the .env file with your API keys
nano .env  # or use your preferred editor
```

## ⚙️ Configuration

### Environment Variables

Edit your `.env` file with the following required variables:

```env
# Required: Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: LangSmith for monitoring
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=form-filling-agent
LANGSMITH_TRACING=true

# Optional: Application settings
ENVIRONMENT=development
MAX_VALIDATION_RETRIES=3
LLM_TIMEOUT=30
```

### Form Configuration

The agent comes pre-configured with a sample form containing four sections:

1. **Personal Information** - Name, email, phone, date of birth
2. **Address Information** - Street address, city, state, zip code, country
3. **Employment Information** - Company, job title, employment status, income
4. **Preferences** - Communication preferences, newsletter subscription, special requests

You can customize these sections by modifying the `FORM_SECTIONS` variable in `agent.py`.

## 🏃‍♂️ Running Locally

### Interactive Mode

Run the agent in interactive mode for testing:

```bash
python agent.py
```

This will start an interactive session where you can test the form filling process:

```
Form Filling Agent - Local Test
========================================
Assistant: Welcome to the Form Filling Assistant! 

I'll help you fill out this form step by step. We have 4 sections to complete:
1. Personal Information
2. Address Information
3. Employment Information
4. Preferences

Let's start with the first section: **Personal Information**
...

User: My name is John Doe and my email is john@example.com
Assistant: Great! I've recorded the following information for Personal Information:
• Full Name: John Doe
• Email: john@example.com

Is this information correct? If you need to make any changes, please let me know. Otherwise, say 'continue' to proceed to validation.
```

### Programmatic Usage

You can also use the agent programmatically:

```python
from agent import app

# Initialize state
initial_state = {
    "form_data": {},
    "current_section": "",
    "section_progress": 0,
    "total_sections": 0,
    "messages": [],
    "form_complete": False,
    "validation_errors": []
}

# Start the form filling process
result = app.invoke(initial_state)
print(result["messages"][-1]["content"])

# Continue with user input
result["messages"].append({"role": "user", "content": "John Doe, john@example.com"})
result = app.invoke(result)
```

## 🌐 Deployment Instructions

### LangGraph Platform Deployment

The agent is configured for deployment on the LangGraph Platform using the included `langgraph.json` configuration.

#### Prerequisites for Deployment

1. **LangGraph Platform Account** - Sign up at [LangGraph Platform](https://langchain-ai.github.io/langgraph/)
2. **LangGraph CLI** - Install the CLI tool:
   ```bash
   pip install langgraph-cli
   ```

#### Deployment Steps

1. **Verify Configuration**
   ```bash
   # Check that all required files are present
   ls -la
   # Should show: agent.py, langgraph.json, requirements.txt, .env.example
   ```

2. **Deploy to LangGraph Platform**
   ```bash
   # Login to LangGraph Platform
   langgraph login

   # Deploy the agent
   langgraph deploy
   ```

3. **Configure Environment Variables**
   - Set your `ANTHROPIC_API_KEY` in the platform's environment configuration
   - Configure any additional variables as needed

4. **Test Deployment**
   ```bash
   # Test the deployed agent
   langgraph invoke form_agent '{"messages": []}'
   ```

### Alternative Deployment Options

#### Docker Deployment

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "langgraph", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

#### Local Server

Run as a local server:

```bash
# Install additional server dependencies
pip install "langgraph[server]"

# Start the server
langgraph serve --port 8000
```

## 📖 Example Form Filling Workflow

Here's a complete example of how the agent processes a form section by section:

### Step 1: Form Initialization

```
Assistant: Welcome to the Form Filling Assistant! 

I'll help you fill out this form step by step. We have 4 sections to complete:
1. Personal Information
2. Address Information  
3. Employment Information
4. Preferences

Let's start with the first section: **Personal Information**

Please provide your basic personal information.

Required fields: full_name, email
All fields: full_name, email, phone, date_of_birth

Please provide the information for this section. You can provide it in any format - I'll help organize it properly.
```

### Step 2: User Input Processing

```
User: Hi, I'm Sarah Johnson. You can reach me at sarah.johnson@email.com or call me at (555) 123-4567. I was born on March 15, 1990.
