"""
Form-filling agent implementation using LangGraph.

This module implements a StateGraph-based agent that can fill out forms section by section,
with human-in-the-loop controls for validation and correction, and persistent memory
using checkpointing.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition

from state import (
    FormFillingState,
    create_initial_state,
    is_form_complete,
    get_next_section,
    get_completion_percentage,
    update_section_data,
    mark_section_complete
)

# Load environment variables
load_dotenv()


# Example form sections for demonstration
DEFAULT_FORM_SECTIONS = [
    "personal_information",
    "contact_details", 
    "employment_information",
    "preferences"
]

# Form field definitions for each section
FORM_FIELDS = {
    "personal_information": {
        "first_name": {"type": "str", "required": True, "description": "First name"},
        "last_name": {"type": "str", "required": True, "description": "Last name"},
        "date_of_birth": {"type": "str", "required": True, "description": "Date of birth (YYYY-MM-DD)"},
        "gender": {"type": "str", "required": False, "description": "Gender (optional)"}
    },
    "contact_details": {
        "email": {"type": "str", "required": True, "description": "Email address"},
        "phone": {"type": "str", "required": True, "description": "Phone number"},
        "address": {"type": "str", "required": True, "description": "Street address"},
        "city": {"type": "str", "required": True, "description": "City"},
        "postal_code": {"type": "str", "required": True, "description": "Postal/ZIP code"}
    },
    "employment_information": {
        "company": {"type": "str", "required": True, "description": "Company name"},
        "position": {"type": "str", "required": True, "description": "Job title/position"},
        "years_experience": {"type": "int", "required": True, "description": "Years of experience"},
        "salary_range": {"type": "str", "required": False, "description": "Salary range (optional)"}
    },
    "preferences": {
        "communication_method": {"type": "str", "required": True, "description": "Preferred communication method (email/phone/mail)"},
        "newsletter": {"type": "bool", "required": False, "description": "Subscribe to newsletter (yes/no)"},
        "special_requirements": {"type": "str", "required": False, "description": "Any special requirements (optional)"}
    }
}


def initialize_llm() -> ChatAnthropic:
    """Initialize the language model."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    
    return ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        temperature=0.1
    )


@tool
def validate_section_data(
    section_name: str,
    field_data: Dict[str, Any],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """
    Validate form data for a specific section and request human confirmation.
    
    Args:
        section_name: Name of the form section
        field_data: Dictionary of field names and values to validate
        tool_call_id: Injected tool call ID for LangGraph
        
    Returns:
        str: Validation result message
    """
    # Get field definitions for this section
    section_fields = FORM_FIELDS.get(section_name, {})
    
    # Validate required fields
    validation_errors = []
    for field_name, field_config in section_fields.items():
        if field_config.get("required", False) and field_name not in field_data:
            validation_errors.append(f"Missing required field: {field_name}")
        elif field_name in field_data:
            # Basic type validation
            value = field_data[field_name]
            expected_type = field_config.get("type", "str")
            
            if expected_type == "int" and not isinstance(value, int):
                try:
                    int(str(value))
                except ValueError:
                    validation_errors.append(f"Field '{field_name}' must be a number")
            elif expected_type == "bool" and not isinstance(value, bool):
                if str(value).lower() not in ["true", "false", "yes", "no", "1", "0"]:
                    validation_errors.append(f"Field '{field_name}' must be yes/no or true/false")
    
    # Prepare data for human review
    review_data = {
        "section": section_name,
        "data": field_data,
        "validation_errors": validation_errors,
        "question": f"Please review the {section_name.replace('_', ' ')} section data. Is this correct?"
    }
    
    # Request human validation
    human_response = interrupt(review_data)
    
    # Process human response
    if human_response.get("approved", "").lower() in ["yes", "y", "true", "1"]:
        # Data approved - update state
        state_update = {
            "form_data": {section_name: field_data},
            "validation_errors": None
        }
        return Command(update=state_update, goto="mark_section_complete")
    else:
        # Data needs correction
        corrections = human_response.get("corrections", {})
        corrected_data = {**field_data, **corrections}
        
        state_update = {
            "form_data": {section_name: corrected_data},
            "validation_errors": human_response.get("feedback", [])
        }
        return Command(update=state_update, goto="process_section")


def start_form_node(state: FormFillingState) -> Dict[str, Any]:
    """
    Initialize the form-filling process.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update with welcome message
    """
    if not state.get("current_section"):
        # Initialize with default sections if not set
        return create_initial_state(
            DEFAULT_FORM_SECTIONS,
            "I'd like to fill out a form."
        )
    
    # Create welcome message
    welcome_message = AIMessage(
        content=f"Welcome! I'll help you fill out this form step by step. "
                f"We have {state['total_sections']} sections to complete: "
                f"{', '.join(state['form_data'].keys())}. "
                f"Let's start with the {state['current_section'].replace('_', ' ')} section."
    )
    
    return {"messages": [welcome_message]}


def process_section_node(state: FormFillingState) -> Dict[str, Any]:
    """
    Process the current form section with LLM assistance.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update with LLM response
    """
    llm = initialize_llm()
    current_section = state["current_section"]
    
    if not current_section:
        return {"messages": [AIMessage(content="No current section to process.")]}
    
    # Get field definitions for current section
    section_fields = FORM_FIELDS.get(current_section, {})
    
    # Create system prompt for this section
    field_descriptions = []
    for field_name, field_config in section_fields.items():
        required_text = " (required)" if field_config.get("required", False) else " (optional)"
        field_descriptions.append(f"- {field_name}: {field_config['description']}{required_text}")
    
    system_prompt = f"""You are a helpful form-filling assistant. You are currently helping the user fill out the "{current_section.replace('_', ' ')}" section of a form.

The fields for this section are:
{chr(10).join(field_descriptions)}

Your task is to:
1. Ask the user for the required information in a conversational way
2. Collect all the necessary field values
3. Once you have collected all the information, use the validate_section_data tool to validate and confirm the data

Be friendly, clear, and help the user understand what information is needed. Ask for one or a few related fields at a time, don't overwhelm them with all fields at once.

Current section: {current_section.replace('_', ' ')}
Progress: {len(state['sections_completed'])}/{state['total_sections']} sections completed ({get_completion_percentage(state):.1f}%)"""
    
    # Prepare messages for LLM
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Get LLM response
    response = llm.invoke(messages)
    
    return {"messages": [response]}


def validation_node(state: FormFillingState) -> Dict[str, Any]:
    """
    Handle validation results and determine next steps.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update based on validation results
    """
    # This node processes the results from the validate_section_data tool
    # The actual validation logic is handled in the tool itself
    return {"messages": [AIMessage(content="Processing validation results...")]}


def mark_section_complete_node(state: FormFillingState) -> Dict[str, Any]:
    """
    Mark the current section as complete and move to next section.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update marking section complete
    """
    current_section = state["current_section"]
    
    if not current_section:
        return {"messages": [AIMessage(content="No section to mark as complete.")]}
    
    # Mark section as complete
    state_update = mark_section_complete(state, current_section)
    
    # Add completion message
    next_section = state_update.get("current_section")
    if next_section:
        completion_message = AIMessage(
            content=f"Great! The {current_section.replace('_', ' ')} section is now complete. "
                   f"Let's move on to the {next_section.replace('_', ' ')} section."
        )
    else:
        completion_message = AIMessage(
            content=f"Excellent! The {current_section.replace('_', ' ')} section is complete. "
                   f"All sections are now finished!"
        )
    
    state_update["messages"] = [completion_message]
    return state_update


def completion_node(state: FormFillingState) -> Dict[str, Any]:
    """
    Handle form completion and provide summary.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update with completion message
    """
    # Create form summary
    summary_parts = ["🎉 Congratulations! Your form has been completed successfully!\n\n**Form Summary:**"]
    
    for section_name, section_data in state["form_data"].items():
        if section_data:  # Only show sections with data
            summary_parts.append(f"\n**{section_name.replace('_', ' ').title()}:**")
            for field_name, value in section_data.items():
                summary_parts.append(f"- {field_name.replace('_', ' ').title()}: {value}")
    
    summary_parts.append(f"\n**Total sections completed:** {len(state['sections_completed'])}/{state['total_sections']}")
    summary_parts.append("**Status:** ✅ Complete")
    
    completion_message = AIMessage(content="\n".join(summary_parts))
    
    return {"messages": [completion_message]}


def should_continue(state: FormFillingState) -> str:
    """
    Determine the next node based on current state.
    
    Args:
        state: Current form-filling state
        
    Returns:
        str: Next node name
    """
    # Check if form is complete
    if is_form_complete(state):
        return "completion"
    
    # Check if we have a current section to process
    if state.get("current_section"):
        return "process_section"
    
    # Default to completion if no current section
    return "completion"


def create_form_filling_graph() -> StateGraph:
    """
    Create and configure the form-filling StateGraph.
    
    Returns:
        StateGraph: Compiled graph ready for execution
    """
    # Initialize the graph
    graph_builder = StateGraph(FormFillingState)
    
    # Add nodes
    graph_builder.add_node("start_form", start_form_node)
    graph_builder.add_node("process_section", process_section_node)
    graph_builder.add_node("validation", validation_node)
    graph_builder.add_node("mark_section_complete", mark_section_complete_node)
    graph_builder.add_node("completion", completion_node)
    
    # Add tool node for validation
    tools = [validate_section_data]
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.add_edge(START, "start_form")
    graph_builder.add_edge("start_form", "process_section")
    
    # Conditional edge from process_section
    graph_builder.add_conditional_edges(
        "process_section",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "completion"
        }
    )
    
    # Edge from tools back to validation
    graph_builder.add_edge("tools", "validation")
    
    # Conditional edge from validation
    graph_builder.add_conditional_edges(
        "validation",
        should_continue,
        {
            "process_section": "process_section",
            "completion": "completion"
        }
    )
    
    # Edge from mark_section_complete
    graph_builder.add_conditional_edges(
        "mark_section_complete",
        should_continue,
        {
            "process_section": "process_section", 
            "completion": "completion"
        }
    )
    
    graph_builder.add_edge("completion", END)
    
    return graph_builder


# Create the graph with memory
def create_agent():
    """Create the form-filling agent with memory."""
    graph_builder = create_form_filling_graph()
    
    # Create memory checkpointer
    memory = MemorySaver()
    
    # Compile the graph with checkpointer
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


# Export the compiled graph as 'app' for deployment compatibility
app = create_agent()


if __name__ == "__main__":
    """
    Simple test of the form-filling agent.
    """
    print("Form-filling agent created successfully!")
    print("Available form sections:", DEFAULT_FORM_SECTIONS)
    print("Use this agent by calling app.stream() or app.invoke() with appropriate config.")
