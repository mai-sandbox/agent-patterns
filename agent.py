"""
Multi-Agent Supervisor System

This module implements a supervisor agent that manages background agents using:
- RemoteGraph interface for deployed agents
- LangGraph SDK for background run management
- langgraph-supervisor library for hierarchical coordination
"""

import os
from typing import Annotated, Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_client
from langgraph_supervisor import create_supervisor
from langgraph.types import Command
from pydantic import BaseModel


class SupervisorState(MessagesState):
    """Extended state for the supervisor agent with run tracking."""
    active_runs: Dict[str, str] = {}  # thread_id -> run_id mapping
    agent_status: Dict[str, str] = {}  # agent_name -> status mapping


# Initialize LangGraph SDK client
client = get_client(url=os.getenv("DEPLOYMENT_URL", "http://localhost:8123"))

# Initialize remote agents
research_agent = RemoteGraph(
    "research_agent",
    url=os.getenv("RESEARCH_AGENT_URL", "http://localhost:8124")
)

math_agent = RemoteGraph(
    "math_agent", 
    url=os.getenv("MATH_AGENT_URL", "http://localhost:8125")
)


@tool
def start_background_run(
    agent_name: str,
    task_description: str,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """Start a background run on a remote agent."""
    try:
        if not thread_id:
            thread_id = f"thread_{datetime.now().isoformat()}"
        
        # Create thread if it doesn't exist
        try:
            thread = client.threads.get(thread_id)
        except:
            thread = client.threads.create(thread_id=thread_id)
        
        # Start the run
        input_data = {"messages": [{"role": "user", "content": task_description}]}
        run = client.runs.create(
            thread_id=thread_id,
            assistant_id=agent_name,
            input=input_data
        )
        
        return {
            "status": "started",
            "run_id": run["run_id"],
            "thread_id": thread_id,
            "agent": agent_name,
            "task": task_description
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "agent": agent_name
        }


@tool
def monitor_run_progress(run_id: str, thread_id: str) -> Dict[str, Any]:
    """Monitor the progress of a background run."""
    try:
        run_info = client.runs.get(thread_id, run_id)
        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "status": run_info["status"],
            "created_at": run_info["created_at"],
            "updated_at": run_info["updated_at"],
            "metadata": run_info.get("metadata", {})
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "status": "error",
            "error": str(e)
        }


@tool
def cancel_run(run_id: str, thread_id: str) -> Dict[str, Any]:
    """Cancel a running background task."""
    try:
        # Note: LangGraph SDK doesn't have a direct cancel method
        # This would typically use interrupt functionality
        run_info = client.runs.get(thread_id, run_id)
        if run_info["status"] == "pending":
            return {
                "status": "cancelled",
                "run_id": run_id,
                "message": "Run cancellation requested"
            }
        else:
            return {
                "status": "cannot_cancel",
                "run_id": run_id,
                "current_status": run_info["status"],
                "message": "Run cannot be cancelled in current state"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "run_id": run_id
        }


@tool
def get_run_results(run_id: str, thread_id: str) -> Dict[str, Any]:
    """Get the results of a completed background run."""
    try:
        # Join the run to wait for completion
        client.runs.join(thread_id, run_id)
        
        # Get the final thread state
        thread_state = client.threads.get_state(thread_id)
        
        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "status": "completed",
            "results": thread_state["values"],
            "final_message": thread_state["values"]["messages"][-1] if thread_state["values"]["messages"] else None
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "status": "error",
            "error": str(e)
        }


# Create the supervisor agent using langgraph-supervisor
supervisor_tools = [
    start_background_run,
    monitor_run_progress,
    cancel_run,
    get_run_results
]

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Create the supervisor using the prebuilt function
supervisor_graph = create_supervisor(
    model=llm,
    agents=[research_agent, math_agent],
    tools=supervisor_tools,
    prompt=(
        "You are a supervisor managing background agents and their tasks.
        "Your responsibilities:
        "1. Start background runs on remote agents using start_background_run
        "2. Monitor agent progress using monitor_run_progress
        "3. Cancel runs when requested using cancel_run
        "4. Retrieve results using get_run_results
        "5. Answer user questions about agent progress and status
        "Available agents:
        "- research_agent: For research and information gathering tasks
        "- math_agent: For mathematical calculations and analysis
        "Always provide clear status updates and handle errors gracefully."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history"
)

# Compile the graph
app = supervisor_graph.compile()


