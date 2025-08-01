import os
import sys
import logging
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ChatNode, SimpleMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_api_key() -> str:
    """Load the OpenAI API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise EnvironmentError("OPENAI_API_KEY not set. Please set it in your .env file.")
    return api_key


def build_graph(api_key: str) -> Any:
    """Build a simple LangGraph with a single ChatNode and memory."""
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    memory = SimpleMemory()
    chat_node = ChatNode(llm=llm, memory=memory)
    graph = StateGraph()
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()


def main() -> None:
    """Main interactive CLI loop for the chatbot."""
    try:
        api_key = load_api_key()
        graph = build_graph(api_key)
    except Exception as e:
        logger.exception(f"Failed to initialize chatbot: {e}")
        sys.exit(1)

    logger.info("Chatbot is ready. Type 'exit' or 'quit' to stop.")
    state: Dict[str, Any] = {"messages": []}
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            state["messages"].append({"role": "user", "content": user_input})
            result = graph.invoke(state)
            messages = result.get("messages", [])
            if messages:
                assistant_msg = messages[-1]["content"]
                print(f"Assistant: {assistant_msg}")
            else:
                print("Assistant: (no response)")
        except KeyboardInterrupt:
            print("
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print("An error occurred. Please try again.")

