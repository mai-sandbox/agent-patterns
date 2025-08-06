# Two-Stage Review Agent using LangGraph

A generic two-stage review system built with LangGraph that implements a ReAct (Reasoning + Acting) agent followed by a review agent for quality assurance. The system automatically iterates until the output meets quality standards or reaches maximum iterations.

## 🌟 Features

- **Generic ReAct Agent**: Accepts any LLM and tools configuration
- **Intelligent Review System**: Automated quality evaluation with customizable criteria
- **Flexible Configuration**: Support for multiple LLM providers (OpenAI, Anthropic, Google)
- **Iterative Improvement**: Automatic feedback loop for response refinement
- **Production Ready**: Deployable with LangGraph deployment infrastructure
- **Type Safe**: Comprehensive type annotations throughout
- **Extensible**: Easy to customize review criteria and iteration limits

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- API key for at least one supported LLM provider

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd agent-patterns
   ```

2. **Create a virtual environment**
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
   # Edit .env with your API keys
   ```

5. **Run the example**
   ```bash
   python agent.py
   ```

## 🔧 Environment Setup

### Required Environment Variables

Choose **one** of the following LLM providers:

#### OpenAI (Recommended)
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
```

#### Anthropic Claude
```bash
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
```

#### Google Gemini
```bash
GOOGLE_API_KEY=your-google-api-key-here
```

### Optional Environment Variables

```bash
# LangSmith for tracing (recommended for debugging)
LANGSMITH_API_KEY=your-langsmith-api-key-here

# Agent configuration
DEFAULT_MAX_ITERATIONS=3
DEFAULT_TEMPERATURE=0.1
```

## 📖 Usage Examples

### Basic Usage

```python
from agent import create_two_stage_review_agent
from langchain_core.messages import HumanMessage

# Create the agent with default configuration
agent = create_two_stage_review_agent()

# Define initial state
initial_state = {
    "messages": [HumanMessage(content="Explain quantum computing in simple terms.")],
    "current_agent": "react",
    "iteration_count": 0,
    "review_decision": None,
    "max_iterations": 3,
    "react_agent_config": {},
    "review_criteria": "The explanation should be clear, accurate, and beginner-friendly."
}

# Run the agent
result = agent.invoke(initial_state)

# Print the final response
for message in result["messages"]:
    if hasattr(message, 'content'):
        print(f"{message.__class__.__name__}: {message.content}")
```

### Custom LLM Configuration

```python
from langchain.chat_models import init_chat_model
from agent import create_two_stage_review_agent

# Use different models for ReAct and Review agents
react_llm = init_chat_model("gpt-4", temperature=0.2)
review_llm = init_chat_model("claude-3-sonnet-20240229", temperature=0.0)

agent = create_two_stage_review_agent(
    react_llm=react_llm,
    review_llm=review_llm,
    max_iterations=5
)
```

### Adding Tools to ReAct Agent

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from agent import create_two_stage_review_agent

# Create tools
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]

# Create agent with tools
agent = create_two_stage_review_agent(
    tools=tools,
    max_iterations=3
)

# Use with a query that benefits from external information
initial_state = {
    "messages": [HumanMessage(content="What are the latest developments in renewable energy?")],
    "current_agent": "react",
    "iteration_count": 0,
    "review_decision": None,
    "max_iterations": 3,
    "react_agent_config": {},
    "review_criteria": "The response should include recent, factual information about renewable energy developments."
}
```

### Custom Review Criteria

```python
from agent import create_two_stage_review_agent

# Define specific review criteria
custom_criteria = """
The response must meet ALL of the following criteria:
1. Factual accuracy - no false information
2. Completeness - addresses all aspects of the question
3. Clarity - easy to understand for the target audience
4. Structure - well-organized with clear sections
5. Actionability - provides practical next steps where applicable
6. Length - between 200-500 words
7. Tone - professional but approachable
"""

agent = create_two_stage_review_agent(
    review_criteria=custom_criteria,
    max_iterations=4
)
```

### Streaming Responses

```python
from agent import create_two_stage_review_agent

agent = create_two_stage_review_agent()

# Stream the agent's execution
for event in agent.stream(initial_state):
    for node_name, node_output in event.items():
        print(f"Node: {node_name}")
        if "messages" in node_output:
            latest_message = node_output["messages"][-1]
            print(f"Content: {latest_message.content[:100]}...")
        print("-" * 50)
```

## 🎛️ Configuration Options

### Agent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `react_llm` | `BaseChatModel` | `gpt-4` | Language model for the ReAct agent |
| `review_llm` | `BaseChatModel` | Same as `react_llm` | Language model for the review agent |
| `tools` | `List[BaseTool]` | `None` | Tools available to the ReAct agent |
| `max_iterations` | `int` | `3` | Maximum number of review iterations |
| `review_criteria` | `str` | Default criteria | Custom review evaluation criteria |

### State Schema

The `ReviewState` includes:

```python
class ReviewState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_agent: Literal["react", "review"]
    iteration_count: int
    review_decision: Optional[Literal["approve", "reject"]]
    max_iterations: int
    react_agent_config: Dict[str, Any]
    review_criteria: str
```

## 🚀 Deployment

### Local Development Server

```bash
# Install LangGraph CLI (if not already installed)
pip install langgraph-cli

# Start the development server
langgraph dev
```

### Production Deployment

The agent is configured for deployment with LangGraph Cloud:

1. **Ensure your `langgraph.json` is properly configured**
2. **Set environment variables in your deployment environment**
3. **Deploy using LangGraph CLI or platform-specific tools**

```bash
# Deploy to LangGraph Cloud
langgraph deploy
```

## 🔍 Troubleshooting

### Common Issues

#### 1. API Key Not Found
```
Error: No API key found for any supported provider
```
**Solution**: Ensure you have set at least one API key in your `.env` file:
```bash
OPENAI_API_KEY=your-key-here
# OR
ANTHROPIC_API_KEY=your-key-here
# OR
GOOGLE_API_KEY=your-key-here
```

#### 2. Import Errors
```
ImportError: No module named 'langgraph'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Maximum Iterations Reached
```
Agent stopped after reaching maximum iterations without approval
```
**Solution**: 
- Increase `max_iterations` parameter
- Refine your `review_criteria` to be more specific
- Check if the ReAct agent has access to necessary tools/information

#### 4. Review Agent Always Rejects
**Solution**:
- Simplify your review criteria
- Check if the criteria are realistic for the given task
- Ensure the ReAct agent has sufficient context/tools

#### 5. Tool Calling Issues
```
Error: Tool 'tool_name' not found
```
**Solution**:
- Ensure tools are properly imported and configured
- Verify tool schemas are valid
- Check that the LLM supports tool calling

### Debug Mode

Enable debug logging by setting environment variables:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

Or programmatically:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Use appropriate temperature settings**:
   - ReAct agent: 0.1-0.3 for focused responses
   - Review agent: 0.0-0.1 for consistent evaluation

2. **Optimize review criteria**:
   - Be specific but not overly restrictive
   - Focus on measurable quality aspects

3. **Set reasonable iteration limits**:
   - 3-5 iterations usually sufficient
   - Higher limits may increase costs without significant quality gains

## 🧪 Testing

### Run the Example

```bash
python agent.py
```

### Custom Test

```python
def test_custom_scenario():
    from agent import create_two_stage_review_agent
    from langchain_core.messages import HumanMessage
    
    agent = create_two_stage_review_agent(max_iterations=2)
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Your test query here")],
        "current_agent": "react",
        "iteration_count": 0,
        "review_decision": None,
        "max_iterations": 2,
        "react_agent_config": {},
        "review_criteria": "Your custom criteria here"
    })
    
    print("Test completed successfully!")
    return result

if __name__ == "__main__":
    test_custom_scenario()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run code quality checks:
   ```bash
   ruff check .
   mypy . --ignore-missing-imports
   ```
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith for Observability](https://smith.langchain.com/)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
3. Open an issue in the repository

---

**Happy building with LangGraph! 🎉**
