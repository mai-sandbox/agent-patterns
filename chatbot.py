import os
import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph



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

    def call_llm(self, state: AgentState):
        messages = state['messages']
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def after_llm(self, state: AgentState):
        return END


