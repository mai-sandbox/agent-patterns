"""
LangGraph Form Filling Agent

A sequential form filling agent that processes forms section by section,
collecting user input and validating data before proceeding to the next section.
"""

import os
from typing import Annotated, Dict, Any, List, Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command


class State(TypedDict):
    """State schema for the form filling agent."""
    form_data: Dict[str, Any]  # Collected form data
    current_section: str  # Current form section being processed
    section_progress: int  # Current section number (0-based)
    total_sections: int  # Total number of sections in the form
    messages: Annotated[List, add_messages]  # Conversation history
    form_complete: bool  # Whether the form is complete
    validation_errors: List[str]  # Any validation errors


# Initialize the LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.1
)

# Define form sections - this could be made configurable
FORM_SECTIONS = [
    {
        "name": "personal_info",
        "title": "Personal Information",
        "fields": ["full_name", "email", "phone", "date_of_birth"],
        "required": ["full_name", "email"],
        "description": "Please provide your basic personal information."
    },
    {
        "name": "address",
        "title": "Address Information", 
        "fields": ["street_address", "city", "state", "zip_code", "country"],
        "required": ["street_address", "city", "state", "zip_code"],
        "description": "Please provide your current address details."
    },
    {
        "name": "employment",
        "title": "Employment Information",
        "fields": ["company_name", "job_title", "employment_status", "annual_income"],
        "required": ["employment_status"],
        "description": "Please provide your employment details."
    },
    {
        "name": "preferences",
        "title": "Preferences",
        "fields": ["communication_preference", "newsletter_subscription", "special_requests"],
        "required": [],
        "description": "Please specify your preferences and any special requests."
    }
]


def start_form(state: State) -> Dict[str, Any]:
    """Initialize the form filling process."""
    current_section = FORM_SECTIONS[0]
    
    welcome_message = f"""
Welcome to the Form Filling Assistant! 

I'll help you fill out this form step by step. We have {len(FORM_SECTIONS)} sections to complete:
{chr(10).join([f"{i+1}. {section['title']}" for i, section in enumerate(FORM_SECTIONS)])}

Let's start with the first section: **{current_section['title']}**

{current_section['description']}

Required fields: {', '.join(current_section['required']) if current_section['required'] else 'None'}
All fields: {', '.join(current_section['fields'])}

Please provide the information for this section. You can provide it in any format - I'll help organize it properly.
"""
    
    return {
        "form_data": {},
        "current_section": current_section["name"],
        "section_progress": 0,
        "total_sections": len(FORM_SECTIONS),
        "messages": [{"role": "assistant", "content": welcome_message}],
        "form_complete": False,
        "validation_errors": []
    }


def process_section(state: State) -> Dict[str, Any]:
    """Process user input for the current section."""
    current_section_info = next(
        (section for section in FORM_SECTIONS if section["name"] == state["current_section"]), 
        None
    )
    
    if not current_section_info:
        return {
            "messages": [{"role": "assistant", "content": "Error: Invalid section. Please restart the form."}],
            "validation_errors": ["Invalid section"]
        }
    
    # Get the latest user message
    user_messages = [msg for msg in state["messages"] if msg.get("role") == "user"]
    if not user_messages:
        return {
            "messages": [{"role": "assistant", "content": "Please provide your information for this section."}]
        }
    
    latest_user_input = user_messages[-1]["content"]
    
    # Use LLM to extract structured data from user input
    extraction_prompt = f"""
You are helping to fill out a form. The user is currently working on the "{current_section_info['title']}" section.

Section fields: {', '.join(current_section_info['fields'])}
Required fields: {', '.join(current_section_info['required']) if current_section_info['required'] else 'None'}

User input: "{latest_user_input}"

Please extract the relevant information and format it as a JSON object with the field names as keys. 
Only include fields that the user has provided information for. 
If a field is not mentioned or unclear, don't include it.

Example format:
{{"full_name": "John Doe", "email": "john@example.com"}}

Return only the JSON object, no other text.
"""
    
    try:
        response = llm.invoke([{"role": "user", "content": extraction_prompt}])
        extracted_data_str = response.content.strip()
        
        # Try to parse the JSON response
        import json
        try:
            extracted_data = json.loads(extracted_data_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract key-value pairs manually
            extracted_data = {}
            lines = extracted_data_str.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().strip('"').lower().replace(' ', '_')
                    value = value.strip().strip('"').strip(',')
                    if key in current_section_info['fields']:
                        extracted_data[key] = value
        
        # Update form data
        updated_form_data = state["form_data"].copy()
        updated_form_data.update(extracted_data)
        
        # Create confirmation message
        confirmation_msg = f"Great! I've recorded the following information for {current_section_info['title']}:\n\n"
        for key, value in extracted_data.items():
            confirmation_msg += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        confirmation_msg += f"\nIs this information correct? If you need to make any changes, please let me know. Otherwise, say 'continue' to proceed to validation."
        
        return {
            "form_data": updated_form_data,
            "messages": [{"role": "assistant", "content": confirmation_msg}]
        }
        
    except Exception as e:
        return {
            "messages": [{"role": "assistant", "content": f"I had trouble processing your input. Please try again with your {current_section_info['title'].lower()} information."}],
            "validation_errors": [f"Processing error: {str(e)}"]
        }


def validate_section(state: State) -> Dict[str, Any]:
    """Validate the current section data."""
    current_section_info = next(
        (section for section in FORM_SECTIONS if section["name"] == state["current_section"]), 
        None
    )
    
    if not current_section_info:
        return {"validation_errors": ["Invalid section"]}
    
    validation_errors = []
    
    # Check required fields
    for required_field in current_section_info["required"]:
        if required_field not in state["form_data"] or not state["form_data"][required_field]:
            validation_errors.append(f"Required field '{required_field.replace('_', ' ').title()}' is missing")
    
    # Basic validation for specific fields
    form_data = state["form_data"]
    
    # Email validation
    if "email" in form_data and form_data["email"]:
        email = form_data["email"]
        if "@" not in email or "." not in email.split("@")[-1]:
            validation_errors.append("Please provide a valid email address")
    
    # Phone validation (basic)
    if "phone" in form_data and form_data["phone"]:
        phone = form_data["phone"]
        digits_only = ''.join(filter(str.isdigit, phone))
        if len(digits_only) < 10:
            validation_errors.append("Please provide a valid phone number")
    
    if validation_errors:
        error_msg = f"Please correct the following issues with your {current_section_info['title']}:\n\n"
        for error in validation_errors:
            error_msg += f"• {error}\n"
        error_msg += "\nPlease provide the corrected information."
        
        return {
            "validation_errors": validation_errors,
            "messages": [{"role": "assistant", "content": error_msg}]
        }
    
    return {"validation_errors": []}


def next_section(state: State) -> Dict[str, Any]:
    """Move to the next section of the form."""
    next_section_index = state["section_progress"] + 1
    
    if next_section_index >= len(FORM_SECTIONS):
        # All sections completed
        return {
            "form_complete": True,
            "messages": [{"role": "assistant", "content": "Excellent! You've completed all sections. Let me prepare your final form submission."}]
        }
    
    next_section_info = FORM_SECTIONS[next_section_index]
    
    progress_msg = f"""
Great! Section {state['section_progress'] + 1} of {len(FORM_SECTIONS)} completed.

Now let's move to the next section: **{next_section_info['title']}**

{next_section_info['description']}

Required fields: {', '.join(next_section_info['required']) if next_section_info['required'] else 'None'}
All fields: {', '.join(next_section_info['fields'])}

Please provide the information for this section.
"""
    
    return {
        "current_section": next_section_info["name"],
        "section_progress": next_section_index,
        "messages": [{"role": "assistant", "content": progress_msg}]
    }


def complete_form(state: State) -> Dict[str, Any]:
    """Complete the form filling process."""
    # Generate a summary of all collected data
    summary_msg = "🎉 **Form Completed Successfully!** 🎉\n\n"
    summary_msg += "Here's a summary of all the information you provided:\n\n"
    
    for section in FORM_SECTIONS:
        section_data = {k: v for k, v in state["form_data"].items() if k in section["fields"]}
        if section_data:
            summary_msg += f"**{section['title']}:**\n"
            for key, value in section_data.items():
                summary_msg += f"• {key.replace('_', ' ').title()}: {value}\n"
            summary_msg += "\n"
    
    summary_msg += "Your form has been successfully completed and is ready for submission!"
    summary_msg += "\n\nThank you for using the Form Filling Assistant!"
    
    return {
        "form_complete": True,
        "messages": [{"role": "assistant", "content": summary_msg}]
    }


def should_validate(state: State) -> Literal["validate_section", "process_section"]:
    """Determine if we should validate or continue processing."""
    # Check if user wants to continue or if they're providing corrections
    user_messages = [msg for msg in state["messages"] if msg.get("role") == "user"]
    if user_messages:
        latest_input = user_messages[-1]["content"].lower()
        if "continue" in latest_input or "yes" in latest_input or "correct" in latest_input:
            return "validate_section"
    
    return "process_section"


def should_proceed(state: State) -> Literal["next_section", "process_section"]:
    """Determine if we should proceed to next section or stay in current one."""
    if state["validation_errors"]:
        return "process_section"
    return "next_section"


def is_form_complete(state: State) -> Literal["complete_form", "process_section"]:
    """Check if the form is complete."""
    if state["form_complete"]:
        return "complete_form"
    return "process_section"


# Create the StateGraph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("start_form", start_form)
graph_builder.add_node("process_section", process_section)
graph_builder.add_node("validate_section", validate_section)
graph_builder.add_node("next_section", next_section)
graph_builder.add_node("complete_form", complete_form)

# Add edges
graph_builder.add_edge(START, "start_form")
graph_builder.add_edge("start_form", "process_section")

# Add conditional edges
graph_builder.add_conditional_edges(
    "process_section",
    should_validate,
    {
        "validate_section": "validate_section",
        "process_section": "process_section"
    }
)

graph_builder.add_conditional_edges(
    "validate_section",
    should_proceed,
    {
        "next_section": "next_section",
        "process_section": "process_section"
    }
)

graph_builder.add_conditional_edges(
    "next_section",
    is_form_complete,
    {
        "complete_form": "complete_form",
        "process_section": "process_section"
    }
)

graph_builder.add_edge("complete_form", END)

# Compile the graph
app = graph_builder.compile()

if __name__ == "__main__":
    # Test the agent locally
    print("Form Filling Agent - Local Test")
    print("=" * 40)
    
    # Initialize state
    initial_state = {
        "form_data": {},
        "current_section": "",
        "section_progress": 0,
        "total_sections": 0,
        "messages": [],
        "form_complete": False,
        "validation_errors": []
    }
    
    # Start the conversation
    result = app.invoke(initial_state)
    print("Assistant:", result["messages"][-1]["content"])
    
    # Interactive loop for testing
    while not result.get("form_complete", False):
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
                
            # Add user message to state
            result["messages"].append({"role": "user", "content": user_input})
            
            # Process the input
            result = app.invoke(result)
            
            # Print assistant response
            assistant_messages = [msg for msg in result["messages"] if msg.get("role") == "assistant"]
            if assistant_messages:
                print("Assistant:", assistant_messages[-1]["content"])
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
