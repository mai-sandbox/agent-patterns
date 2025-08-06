"""
LangGraph Form Filling Agent

A section-by-section form filling agent that processes forms systematically,
validating each section before moving to the next.
"""

from typing import Dict, List, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# State Schema
class FormState(TypedDict):
    """State schema for the form filling workflow."""
    current_section: str
    form_data: Dict[str, Any]
    sections_completed: List[str]
    form_structure: Dict[str, Any]
    messages: List[Any]
    validation_errors: List[str]
    is_complete: bool

# Initialize the LLM
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022"
)

def analyze_form_structure(state: FormState) -> Dict[str, Any]:
    """
    Analyze the incoming form structure and initialize the workflow.
    
    Args:
        state: Current form state
        
    Returns:
        Updated state with form structure analysis
    """
    # Extract form structure from messages or initialize default
    if not state.get("form_structure"):
        # Default form structure for demonstration
        form_structure = {
            "personal_info": {
                "fields": ["name", "email", "phone"],
                "required": ["name", "email"],
                "description": "Personal information section"
            },
            "address": {
                "fields": ["street", "city", "state", "zip"],
                "required": ["street", "city", "state"],
                "description": "Address information section"
            },
            "preferences": {
                "fields": ["newsletter", "notifications", "language"],
                "required": [],
                "description": "User preferences section"
            }
        }
    else:
        form_structure = state["form_structure"]
    
    # Initialize form data structure
    form_data: Dict[str, Dict[str, Any]] = {}
    for section_name, section_info in form_structure.items():
        form_data[section_name] = {}
        for field in section_info["fields"]:
            form_data[section_name][field] = None
    
    # Determine first section to process
    first_section = list(form_structure.keys())[0] if form_structure else "personal_info"
    
    return {
        "form_structure": form_structure,
        "form_data": form_data,
        "current_section": first_section,
        "sections_completed": [],
        "validation_errors": [],
        "is_complete": False
    }

def fill_current_section(state: FormState) -> Dict[str, Any]:
    """
    Fill out the current section of the form using the LLM.
    
    Args:
        state: Current form state
        
    Returns:
        Updated state with filled section data
    """
    current_section = state["current_section"]
    section_info = state["form_structure"][current_section]
    
    # In a real implementation, you would:
    # 1. Create a prompt for the LLM based on section info and conversation context
    # 2. Get LLM response for form filling
    # 3. Parse the LLM response to extract field values
    # For demonstration, we'll simulate filling some fields
    filled_data = {}
    for field in section_info["fields"]:
        if field in ["name", "email"]:
            filled_data[field] = f"sample_{field}"  # Placeholder values
        else:
            filled_data[field] = None  # Needs user input
    
    # Update form data for current section
    updated_form_data = state["form_data"].copy()
    updated_form_data[current_section] = filled_data
    
    return {
        "form_data": updated_form_data,
        "messages": state["messages"] + [AIMessage(content=f"Filled section: {current_section}")]
    }

def validate_section(state: FormState) -> Dict[str, Any]:
    """
    Validate the current section for completeness and correctness.
    
    Args:
        state: Current form state
        
    Returns:
        Updated state with validation results
    """
    current_section = state["current_section"]
    section_info = state["form_structure"][current_section]
    section_data = state["form_data"][current_section]
    
    validation_errors = []
    
    # Check required fields
    for required_field in section_info["required"]:
        if not section_data.get(required_field):
            validation_errors.append(f"Missing required field: {required_field} in {current_section}")
    
    # Basic validation for email format (example)
    if "email" in section_data and section_data["email"]:
        email = section_data["email"]
        if "@" not in email or "." not in email:
            validation_errors.append(f"Invalid email format: {email}")
    
    # If validation passes, mark section as completed
    sections_completed = state["sections_completed"].copy()
    if not validation_errors and current_section not in sections_completed:
        sections_completed.append(current_section)
    
    return {
        "validation_errors": validation_errors,
        "sections_completed": sections_completed
    }

def determine_next_section(state: FormState) -> str:
    """
    Determine the next section to process or end the workflow.
    
    Args:
        state: Current form state
        
    Returns:
        Name of next node to execute
    """
    # If there are validation errors, stay in current section
    if state["validation_errors"]:
        return "fill_current_section"
    
    # Get all sections and find next unprocessed section
    all_sections = list(state["form_structure"].keys())
    completed_sections = state["sections_completed"]
    
    # Find next section to process
    for section in all_sections:
        if section not in completed_sections:
            return "move_to_next_section"
    
    # All sections completed
    return "complete_form"

def move_to_next_section(state: FormState) -> Dict[str, Any]:
    """
    Move to the next section in the form.
    
    Args:
        state: Current form state
        
    Returns:
        Updated state with next section set as current
    """
    all_sections = list(state["form_structure"].keys())
    completed_sections = state["sections_completed"]
    
    # Find next uncompleted section
    next_section: str | None = None
    for section in all_sections:
        if section not in completed_sections:
            next_section = section
            break
    
    if next_section:
        return {
            "current_section": next_section,
            "validation_errors": []  # Clear previous validation errors
        }
    else:
        return {"is_complete": True}

def complete_form(state: FormState) -> Dict[str, Any]:
    """
    Complete the form filling process.
    
    Args:
        state: Current form state
        
    Returns:
        Final state with completion status
    """
    return {
        "is_complete": True,
        "messages": state["messages"] + [AIMessage(content="Form filling completed successfully!")]
    }

def should_continue(state: FormState) -> str:
    """
    Routing function to determine next step in the workflow.
    
    Args:
        state: Current form state
        
    Returns:
        Next node name to execute
    """
    # If form is complete, end workflow
    if state.get("is_complete", False):
        return END
    
    # If there are validation errors, retry filling current section
    if state.get("validation_errors"):
        return "fill_current_section"
    
    # Check if current section is completed
    current_section = state.get("current_section")
    if current_section and current_section in state.get("sections_completed", []):
        # Move to next section or complete
        all_sections = list(state["form_structure"].keys())
        completed_sections = state["sections_completed"]
        
        if len(completed_sections) >= len(all_sections):
            return "complete_form"
        else:
            return "move_to_next_section"
    
    # Continue with current section
    return "validate_section"

# Create the StateGraph
workflow = StateGraph(FormState)

# Add nodes
workflow.add_node("analyze_form_structure", analyze_form_structure)
workflow.add_node("fill_current_section", fill_current_section)
workflow.add_node("validate_section", validate_section)
workflow.add_node("move_to_next_section", move_to_next_section)
workflow.add_node("complete_form", complete_form)

# Add edges
workflow.add_edge(START, "analyze_form_structure")
workflow.add_edge("analyze_form_structure", "fill_current_section")
workflow.add_edge("fill_current_section", "validate_section")
workflow.add_edge("move_to_next_section", "fill_current_section")
workflow.add_edge("complete_form", END)

# Add conditional edges
workflow.add_conditional_edges(
    "validate_section",
    should_continue,
    {
        "fill_current_section": "fill_current_section",
        "move_to_next_section": "move_to_next_section",
        "complete_form": "complete_form",
        END: END
    }
)

# Compile the graph - REQUIRED: Export as 'app' for deployment
app = workflow.compile()

# Optional: Add debugging/visualization support
if __name__ == "__main__":
    # Test the graph with sample input
    sample_input: FormState = {
        "messages": [HumanMessage(content="I need to fill out a form with my information.")],
        "current_section": "",
        "form_data": {},
        "sections_completed": [],
        "form_structure": {},
        "validation_errors": [],
        "is_complete": False
    }
    
    try:
        result = app.invoke(sample_input)
        print("Form filling workflow completed successfully!")
        print(f"Final state: {result}")
    except Exception as e:
        print(f"Error running workflow: {e}")








