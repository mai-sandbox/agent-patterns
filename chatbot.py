import operator
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


SYSTEM_MESSAGE = "You are a helpful AI assistant. Answer the user's questions to the best of your ability."


class Agent:
    def __init__(self, model, checkpointer):
        self.model = model
        self.checkpointer = checkpointer

    def get_runnable(self):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_conditional_edge(
            "llm",
            self.after_llm,
        )
        graph.set_entry_point("llm")
        runnable = graph.compile(checkpointer=self.checkpointer)
        return runnable

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_MESSAGE)] + (messages or [])
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def after_llm(self, state: AgentState):
        return END

if __name__ == "__main__":
    model = ChatAnthropic(model_name="claude-3-haiku-20240307")
    checkpointer = SqliteSaver.from_conn_string(":memory:")
    agent = Agent(model, checkpointer)
    runnable = agent.get_runnable()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        try:
            response = runnable.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            if response and response['messages']:
                assistant_message = response['messages'][-1]
                if hasattr(assistant_message, 'content'):
                    print(f"Assistant: {assistant_message.content}")
                else:
                    print(f"Assistant: {assistant_message}")
        except Exception as e:
            print(f"An error occurred: {e}")



