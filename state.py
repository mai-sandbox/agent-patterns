"""
State schema for the form-filling agent.

This module defines the state structure used by the LangGraph form-filling agent
to track form progress, data, and conversation history across sections.
"""

from typing import Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class FormFillingState(TypedDict):
    """
    State schema for the form-filling agent.
    
    This state tracks the complete form-filling process including:
    - Form data collected across all sections
    - Current section being processed
    - Progress tracking (completed sections)
    - Conversation messages with the user
    """
    
    # Messages list with automatic message appending
    # The add_messages reducer ensures new messages are appended rather than overwritten
    messages: Annotated[List[Any], add_messages]
    
    # Form data collected across all sections
    # Structure: {"section_name": {"field_name": "value", ...}, ...}
    form_data: Dict[str, Dict[str, Any]]
    
    # Current section being processed (e.g., "personal_info", "contact_details")
    current_section: str
    
    # List of completed section names
    sections_completed: List[str]
    
    # Total number of sections in the form
    total_sections: int
    
    # Optional: Current field being processed within a section
    current_field: Optional[str]
    
    # Optional: Validation errors for the current section
    validation_errors: Optional[List[str]]
    
    # Optional: User preferences or configuration
    user_preferences: Optional[Dict[str, Any]]


def create_initial_state(
    form_sections: List[str],
    user_message: str = "I'd like to fill out a form."
) -> FormFillingState:
    """
    Create an initial state for the form-filling process.
    
    Args:
        form_sections: List of section names that make up the form
        user_message: Initial user message to start the conversation
        
    Returns:
        FormFillingState: Initial state with empty form data and first section set
    """
    return FormFillingState(
        messages=[{"role": "user", "content": user_message}],
        form_data={section: {} for section in form_sections},
        current_section=form_sections[0] if form_sections else "",
        sections_completed=[],
        total_sections=len(form_sections),
        current_field=None,
        validation_errors=None,
        user_preferences=None
    )


def is_form_complete(state: FormFillingState) -> bool:
    """
    Check if the form filling process is complete.
    
    Args:
        state: Current form-filling state
        
    Returns:
        bool: True if all sections are completed, False otherwise
    """
    return len(state["sections_completed"]) >= state["total_sections"]


def get_next_section(state: FormFillingState) -> Optional[str]:
    """
    Get the next section to process based on current state.
    
    Args:
        state: Current form-filling state
        
    Returns:
        Optional[str]: Next section name, or None if all sections are complete
    """
    all_sections = list(state["form_data"].keys())
    completed = state["sections_completed"]
    
    for section in all_sections:
        if section not in completed:
            return section
    
    return None


def get_completion_percentage(state: FormFillingState) -> float:
    """
    Calculate the completion percentage of the form.
    
    Args:
        state: Current form-filling state
        
    Returns:
        float: Completion percentage (0.0 to 100.0)
    """
    if state["total_sections"] == 0:
        return 100.0
    
    return (len(state["sections_completed"]) / state["total_sections"]) * 100.0


def validate_state_transition(
    current_state: FormFillingState,
    new_section: str
) -> bool:
    """
    Validate if transitioning to a new section is allowed.
    
    Args:
        current_state: Current form-filling state
        new_section: Section to transition to
        
    Returns:
        bool: True if transition is valid, False otherwise
    """
    # Check if the new section exists in the form
    if new_section not in current_state["form_data"]:
        return False
    
    # Allow transitioning to any incomplete section
    if new_section not in current_state["sections_completed"]:
        return True
    
    # Allow revisiting completed sections for corrections
    return True


def update_section_data(
    state: FormFillingState,
    section: str,
    field_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update form data for a specific section.
    
    Args:
        state: Current form-filling state
        section: Section name to update
        field_data: Dictionary of field names and values to update
        
    Returns:
        Dict[str, Any]: State update dictionary for LangGraph
    """
    # Create a copy of the current form data
    updated_form_data = state["form_data"].copy()
    
    # Update the specific section
    if section in updated_form_data:
        updated_form_data[section].update(field_data)
    else:
        updated_form_data[section] = field_data.copy()
    
    return {"form_data": updated_form_data}


def mark_section_complete(
    state: FormFillingState,
    section: str
) -> Dict[str, Any]:
    """
    Mark a section as completed and update the current section.
    
    Args:
        state: Current form-filling state
        section: Section name to mark as complete
        
    Returns:
        Dict[str, Any]: State update dictionary for LangGraph
    """
    # Add section to completed list if not already there
    sections_completed = state["sections_completed"].copy()
    if section not in sections_completed:
        sections_completed.append(section)
    
    # Get the next section to process
    next_section = get_next_section({
        **state,
        "sections_completed": sections_completed
    })
    
    return {
        "sections_completed": sections_completed,
        "current_section": next_section or "",
        "current_field": None,
        "validation_errors": None
    }
