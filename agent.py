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
    Enhanced validation tool with comprehensive error handling and retry logic.
    
    Args:
        section_name: Name of the form section
        field_data: Dictionary of field names and values to validate
        tool_call_id: Injected tool call ID for LangGraph
        
    Returns:
        str: Validation result message
    """
    try:
        # Get field definitions for this section
        section_fields = FORM_FIELDS.get(section_name, {})
        
        if not section_fields:
            return Command(update={
                "validation_errors": [f"Unknown section: {section_name}"]
            }, goto="process_section")
        
        # Enhanced validation with detailed error messages
        validation_errors = []
        warnings = []
        
        # Check required fields
        for field_name, field_config in section_fields.items():
            if field_config.get("required", False):
                if field_name not in field_data or not field_data[field_name]:
                    validation_errors.append(f"Missing required field: {field_name} ({field_config['description']})")
        
        # Validate field types and formats
        for field_name, value in field_data.items():
            if field_name in section_fields:
                field_config = section_fields[field_name]
                expected_type = field_config.get("type", "str")
                
                # Type validation with conversion attempts
                if expected_type == "int":
                    try:
                        int_value = int(str(value))
                        if int_value < 0 and field_name == "years_experience":
                            validation_errors.append(f"Years of experience cannot be negative")
                        elif int_value > 100 and field_name == "years_experience":
                            warnings.append(f"Years of experience seems unusually high: {int_value}")
                    except ValueError:
                        validation_errors.append(f"Field '{field_name}' must be a valid number, got: {value}")
                
                elif expected_type == "bool":
                    if str(value).lower() not in ["true", "false", "yes", "no", "1", "0"]:
                        validation_errors.append(f"Field '{field_name}' must be yes/no or true/false, got: {value}")
                
                elif expected_type == "str":
                    # Email validation
                    if field_name == "email" and "@" not in str(value):
                        validation_errors.append(f"Email address appears invalid: {value}")
                    # Phone validation (basic)
                    elif field_name == "phone" and len(str(value).replace(" ", "").replace("-", "").replace("(", "").replace(")", "")) < 10:
                        validation_errors.append(f"Phone number appears too short: {value}")
                    # Postal code validation (basic)
                    elif field_name == "postal_code" and len(str(value)) < 5:
                        validation_errors.append(f"Postal code appears invalid: {value}")
        
        # Calculate completion score
        total_fields = len(section_fields)
        completed_fields = len([f for f in field_data.values() if f])
        completion_score = (completed_fields / total_fields) * 100 if total_fields > 0 else 0
        
        # Prepare enhanced data for human review
        review_data = {
            "section": section_name,
            "section_display_name": section_name.replace('_', ' ').title(),
            "data": field_data,
            "validation_errors": validation_errors,
            "warnings": warnings,
            "completion_score": completion_score,
            "total_fields": total_fields,
            "completed_fields": completed_fields,
            "question": f"Please review the {section_name.replace('_', ' ')} section data. Is this correct?",
            "retry_count": field_data.get("_retry_count", 0)
        }
        
        # Request human validation with enhanced context
        human_response = interrupt(review_data)
        
        # Process human response with retry logic
        retry_count = field_data.get("_retry_count", 0)
        
        if human_response.get("approved", "").lower() in ["yes", "y", "true", "1", "approve", "correct"]:
            # Data approved - clean up retry count and update state
            clean_data = {k: v for k, v in field_data.items() if not k.startswith("_")}
            state_update = {
                "form_data": {section_name: clean_data},
                "validation_errors": None,
                "current_field": None
            }
            return Command(update=state_update, goto="mark_section_complete")
        
        elif human_response.get("action", "").lower() == "retry" and retry_count < 3:
            # Retry with corrections
            corrections = human_response.get("corrections", {})
            feedback = human_response.get("feedback", [])
            
            # Merge corrections with existing data
            corrected_data = {**field_data, **corrections}
            corrected_data["_retry_count"] = retry_count + 1
            
            state_update = {
                "form_data": {section_name: corrected_data},
                "validation_errors": feedback,
                "current_field": human_response.get("focus_field")
            }
            return Command(update=state_update, goto="process_section")
        
        elif human_response.get("action", "").lower() == "skip_section":
            # Skip this section (mark as complete but with warnings)
            state_update = {
                "form_data": {section_name: field_data},
                "validation_errors": [f"Section skipped by user with {len(validation_errors)} validation errors"],
                "current_field": None
            }
            return Command(update=state_update, goto="mark_section_complete")
        
        else:
            # Too many retries or user wants to restart section
            if retry_count >= 3:
                feedback = ["Maximum retry attempts reached. Please restart this section."]
            else:
                feedback = human_response.get("feedback", ["User requested section restart"])
            
            # Reset section data
            state_update = {
                "form_data": {section_name: {}},
                "validation_errors": feedback,
                "current_field": None
            }
            return Command(update=state_update, goto="process_section")
            
    except Exception as e:
        # Comprehensive error handling for validation tool
        error_msg = f"Validation error: {str(e)}"
        state_update = {
            "validation_errors": [error_msg],
            "current_field": None
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
    Enhanced with error handling and retry logic.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update with LLM response
    """
    try:
        llm = initialize_llm()
        current_section = state["current_section"]
        
        if not current_section:
            return {
                "messages": [AIMessage(content="Error: No current section to process. Please restart the form.")],
                "validation_errors": ["No current section defined"]
            }
        
        # Get field definitions for current section
        section_fields = FORM_FIELDS.get(current_section, {})
        
        if not section_fields:
            return {
                "messages": [AIMessage(content=f"Error: Unknown section '{current_section}'. Please contact support.")],
                "validation_errors": [f"Unknown section: {current_section}"]
            }
        
        # Check for existing data in this section
        existing_data = state.get("form_data", {}).get(current_section, {})
        validation_errors = state.get("validation_errors", [])
        
        # Create system prompt for this section
        field_descriptions = []
        for field_name, field_config in section_fields.items():
            required_text = " (required)" if field_config.get("required", False) else " (optional)"
            current_value = existing_data.get(field_name, "")
            value_text = f" [Current: {current_value}]" if current_value else ""
            field_descriptions.append(f"- {field_name}: {field_config['description']}{required_text}{value_text}")
        
        # Add error context if there are validation errors
        error_context = ""
        if validation_errors:
            error_context = f"\n\nPrevious validation errors to address:\n" + "\n".join(f"- {error}" for error in validation_errors)
        
        # Calculate progress
        progress_pct = get_completion_percentage(state)
        completed_count = len(state['sections_completed'])
        total_count = state['total_sections']
        
        system_prompt = f"""You are a helpful form-filling assistant. You are currently helping the user fill out the "{current_section.replace('_', ' ')}" section of a form.

The fields for this section are:
{chr(10).join(field_descriptions)}

Your task is to:
1. Ask the user for the required information in a conversational way
2. Collect all the necessary field values
3. Once you have collected all the information, use the validate_section_data tool to validate and confirm the data
4. If there are validation errors, help the user correct them

Be friendly, clear, and help the user understand what information is needed. Ask for one or a few related fields at a time, don't overwhelm them with all fields at once.

Current section: {current_section.replace('_', ' ')}
Progress: {completed_count}/{total_count} sections completed ({progress_pct:.1f}%)
Remaining sections: {', '.join([s for s in state.get('form_data', {}).keys() if s not in state.get('sections_completed', [])])}{error_context}"""
        
        # Prepare messages for LLM
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        # Get LLM response with error handling
        try:
            response = llm.invoke(messages)
            return {"messages": [response]}
        except Exception as llm_error:
            error_message = AIMessage(
                content=f"I'm experiencing technical difficulties. Please try again. "
                       f"If the problem persists, please contact support. (Error: {str(llm_error)[:100]})"
            )
            return {
                "messages": [error_message],
                "validation_errors": [f"LLM error: {str(llm_error)}"]
            }
            
    except Exception as e:
        # Comprehensive error handling
        error_message = AIMessage(
            content=f"An unexpected error occurred while processing this section. "
                   f"Please try again or contact support if the issue persists."
        )
        return {
            "messages": [error_message],
            "validation_errors": [f"Processing error: {str(e)}"]
        }


def validation_node(state: FormFillingState) -> Dict[str, Any]:
    """
    Enhanced validation node with comprehensive error handling and progress tracking.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update based on validation results
    """
    try:
        current_section = state.get("current_section", "")
        validation_errors = state.get("validation_errors", [])
        
        # Check if there are validation errors to handle
        if validation_errors:
            error_count = len(validation_errors)
            error_summary = f"Found {error_count} validation issue{'s' if error_count != 1 else ''} in the {current_section.replace('_', ' ')} section."
            
            # Create detailed error message
            error_details = "\n".join(f"• {error}" for error in validation_errors)
            
            validation_message = AIMessage(
                content=f"⚠️ {error_summary}\n\nIssues to address:\n{error_details}\n\n"
                       f"Please provide the corrected information, and I'll help you fix these issues."
            )
            
            return {
                "messages": [validation_message],
                "current_field": None  # Reset current field focus
            }
        
        # No validation errors - processing successful
        progress_pct = get_completion_percentage(state)
        completed_count = len(state.get('sections_completed', []))
        total_count = state.get('total_sections', 0)
        
        success_message = AIMessage(
            content=f"✅ Validation successful! Processing complete for the {current_section.replace('_', ' ')} section.\n"
                   f"Progress: {completed_count}/{total_count} sections completed ({progress_pct:.1f}%)"
        )
        
        return {"messages": [success_message]}
        
    except Exception as e:
        # Error handling for validation node
        error_message = AIMessage(
            content=f"An error occurred during validation processing. Please try again. "
                   f"If the issue persists, please contact support."
        )
        return {
            "messages": [error_message],
            "validation_errors": [f"Validation node error: {str(e)}"]
        }


def mark_section_complete_node(state: FormFillingState) -> Dict[str, Any]:
    """
    Enhanced section completion node with comprehensive progress tracking and error handling.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Dict[str, Any]: State update marking section complete
    """
    try:
        current_section = state.get("current_section", "")
        
        if not current_section:
            error_message = AIMessage(
                content="⚠️ Error: No section to mark as complete. Please restart the form process."
            )
            return {
                "messages": [error_message],
                "validation_errors": ["No current section to complete"]
            }
        
        # Validate that the section has required data
        section_data = state.get("form_data", {}).get(current_section, {})
        section_fields = FORM_FIELDS.get(current_section, {})
        
        # Check for missing required fields
        missing_required = []
        for field_name, field_config in section_fields.items():
            if field_config.get("required", False) and not section_data.get(field_name):
                missing_required.append(field_name)
        
        if missing_required:
            error_message = AIMessage(
                content=f"⚠️ Cannot complete {current_section.replace('_', ' ')} section. "
                       f"Missing required fields: {', '.join(missing_required)}. "
                       f"Please provide the missing information."
            )
            return {
                "messages": [error_message],
                "validation_errors": [f"Missing required fields: {', '.join(missing_required)}"]
            }
        
        # Mark section as complete using the state utility function
        state_update = mark_section_complete(state, current_section)
        
        # Calculate updated progress
        new_completed_count = len(state_update.get("sections_completed", []))
        total_count = state.get("total_sections", 0)
        progress_pct = (new_completed_count / total_count) * 100 if total_count > 0 else 100
        
        # Get next section info
        next_section = state_update.get("current_section")
        
        # Create detailed completion message
        if next_section:
            # More sections to go
            remaining_sections = [s for s in state.get('form_data', {}).keys() 
                                if s not in state_update.get("sections_completed", [])]
            
            completion_message = AIMessage(
                content=f"✅ Excellent! The **{current_section.replace('_', ' ').title()}** section is now complete!\n\n"
                       f"📊 **Progress Update:**\n"
                       f"• Completed: {new_completed_count}/{total_count} sections ({progress_pct:.1f}%)\n"
                       f"• Next: {next_section.replace('_', ' ').title()}\n"
                       f"• Remaining: {', '.join([s.replace('_', ' ').title() for s in remaining_sections])}\n\n"
                       f"🚀 Let's continue with the {next_section.replace('_', ' ')} section!"
            )
        else:
            # All sections complete
            completion_message = AIMessage(
                content=f"🎉 **Fantastic!** The **{current_section.replace('_', ' ').title()}** section is complete!\n\n"
                       f"📊 **Final Progress:**\n"
                       f"• All {total_count} sections completed (100%)\n"
                       f"• Form is ready for final review!\n\n"
                       f"✨ Proceeding to form completion and summary..."
            )
        
        # Add completion timestamp and metadata
        state_update["messages"] = [completion_message]
        state_update["validation_errors"] = None  # Clear any previous errors
        
        # Add completion metadata
        if "form_data" not in state_update:
            state_update["form_data"] = state.get("form_data", {})
        
        # Add completion timestamp to the section data
        if current_section in state_update["form_data"]:
            state_update["form_data"][current_section]["_completed_at"] = "completed"
        
        return state_update
        
    except Exception as e:
        # Comprehensive error handling
        error_message = AIMessage(
            content=f"⚠️ An error occurred while completing the section. "
                   f"Please try again or contact support if the issue persists."
        )
        return {
            "messages": [error_message],
            "validation_errors": [f"Section completion error: {str(e)}"]
        }


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




