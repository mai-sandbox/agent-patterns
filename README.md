# LangGraph Agent Interaction & Feedback App

A Streamlit application for interacting with LangGraph agents deployed using `langgraph dev`. This app allows you to run agents with custom configurations, stream responses in real-time, collect feedback, and rerun with updated configurations.

## Features

- 🔗 **Connect to LangGraph Agents**: Connect to agents deployed with `langgraph dev`
- ⚙️ **Custom Configuration**: Configure agents with JSON-based settings
- 📡 **Real-time Streaming**: Stream agent responses as they are generated
- 📝 **Feedback Collection**: Provide feedback on agent performance
- 🔄 **Smart Rerun**: Automatically update configuration based on feedback and rerun
- 📊 **Run History**: Track all interactions and configurations
- 📱 **Responsive UI**: Clean, intuitive interface with sidebar configuration

## Prerequisites

1. **Python 3.8+**
2. **LangGraph Agent**: A LangGraph agent deployed and running with `langgraph dev`
3. **Dependencies**: Install from requirements.txt

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Deploy your LangGraph agent** (in a separate terminal):
   ```bash
   langgraph dev
   ```
   This will start your agent at `http://localhost:2024` by default.

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Usage

### 1. Connect to Your Agent

1. In the sidebar, enter your agent URL (default: `http://localhost:2024`)
2. Enter your agent name (default: `agent` - this should match your `langgraph.json` configuration)
3. Click "Test Connection" to verify connectivity

### 2. Configure Your Agent

1. Use the "Agent Configuration" section in the sidebar
2. Enter configuration as JSON format, for example:
   ```json
   {
     "configurable": {
       "temperature": 0.7,
       "max_tokens": 1000,
       "user_id": "user_123"
     }
   }
   ```

### 3. Run Your Agent

1. Enter your message in the main text area
2. Click "Run Agent" to start the interaction
3. Watch the response stream in real-time

### 4. Provide Feedback

1. After the agent completes, use the feedback box on the right
2. Provide specific feedback, such as:
   - "Be more concise"
   - "Provide more details"
   - "Use temperature 0.3"
   - "Be more creative"
3. Click "Submit Feedback" to update the configuration

### 5. Rerun with Updated Configuration

1. After submitting feedback, the "Rerun" button will appear
2. Click "Rerun Agent" to run the same input with the updated configuration
3. Compare the results to see the improvement

## Configuration Examples

### Basic Configuration
```json
{
  "configurable": {
    "temperature": 0.7,
    "user_id": "user_123"
  }
}
```

### Advanced Configuration
```json
{
  "configurable": {
    "temperature": 0.8,
    "max_tokens": 2000,
    "user_id": "user_123",
    "system_prompt": "You are a helpful assistant",
    "verbose": true,
    "model_name": "gpt-4"
  }
}
```

## Feedback Processing

The app automatically processes feedback to update configurations:

- **Temperature**: "temperature 0.5" → sets temperature to 0.5
- **Verbosity**: "be more detailed" → sets verbose to true
- **Conciseness**: "be more concise" → sets verbose to false
- **General feedback**: Added as context for future runs

## Troubleshooting

### Connection Issues

1. **Agent not running**: Make sure your LangGraph agent is running with `langgraph dev`
2. **Wrong URL**: Verify the agent URL (default: `http://localhost:2024`)
3. **Wrong agent name**: Check your `langgraph.json` file for the correct agent name
4. **Port conflicts**: If using a different port, update the URL accordingly

### Configuration Issues

1. **Invalid JSON**: Use the JSON validator in the sidebar to check syntax
2. **Configuration not applied**: Make sure to use the "configurable" key for runtime configuration
3. **Agent errors**: Check the agent logs for configuration-related errors

### Streaming Issues

1. **No response**: Check agent logs for errors
2. **Partial responses**: Verify network connectivity
3. **Slow streaming**: Check agent performance and network latency

## Development

### Project Structure
```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Functions

- `check_agent_connection()`: Tests connectivity to the LangGraph agent
- `initialize_client()`: Creates and configures the LangGraph SDK client
- `stream_agent_response()`: Handles real-time streaming of agent responses
- `update_configuration_from_feedback()`: Processes feedback to update configuration

### Extending the App

1. **Custom feedback processing**: Modify `update_configuration_from_feedback()` to handle specific feedback patterns
2. **Additional streaming modes**: Experiment with different `stream_mode` options
3. **Enhanced UI**: Add more configuration options or visualization features
4. **Persistence**: Add database storage for run history and configurations

## Dependencies

- **streamlit**: Web application framework
- **langgraph-sdk**: LangGraph SDK for agent interaction
- **requests**: HTTP client for health checks
- **python-dotenv**: Environment variable management
- **aiohttp**: Async HTTP operations
- **json5**: Enhanced JSON parsing

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Support

For issues related to:
- **This app**: Create an issue in this repository
- **LangGraph**: Check the [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- **Streamlit**: Check the [Streamlit documentation](https://docs.streamlit.io/)
