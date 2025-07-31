import os
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


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

