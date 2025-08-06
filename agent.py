"""
LangGraph Form Filling Agent

A section-by-section form filling agent that processes forms intelligently,
validates input, and guides users through completion.
"""

import os
from typing import Annotated, Dict, List, Literal, Optional, Union
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


# Custom State class using TypedDict to track form progress
class FormState(TypedDict):
    """State schema for the form filling agent."""
    # Messages with add_messages reducer for conversation history
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Current section being processed
    current_section: str
    
    # Form data collected so far
    form_data: Dict[str, str]
    
    # List of completed sections
    completed_sections: List[str]
    
    # Available sections in the form
    available_sections: List[str]
    
    # Current user input for processing
    user_input: Optional[str]
    
    # Validation errors if any
    validation_errors: List[str]


# Initialize the LLM (following Anthropic preference)
def get_llm() -> ChatAnthropic:
    """Initialize and return the Anthropic LLM."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    return ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        anthropic_api_key=api_key,
        temperature=0.1
    )


# Node functions for form processing
def start_form_node(state: FormState) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
    """Initialize the form filling process."""
    # Define the sections of the form
    sections = [
        "personal_info",
        "contact_details", 
        "employment_info",
        "preferences"
    ]
    
    system_message = SystemMessage(
        content="""You are a helpful form filling assistant. You will guide users through 
        filling out a form section by section. The form has the following sections:
        
        1. Personal Information (name, date of birth, address)
        2. Contact Details (email, phone number)
        3. Employment Information (job title, company, experience)
        4. Preferences (communication preferences, interests)
        
        Start by greeting the user and explaining the process. Ask them to begin with 
        the first section: Personal Information."""
    )
    
    return {
        "messages": [system_message],
        "current_section": "personal_info",
        "available_sections": sections,
        "completed_sections": [],
        "form_data": {},
        "validation_errors": []
    }


def process_section_node(state: FormState) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
    """Process the current section with LLM assistance."""
    llm = get_llm()
    
    current_section = state["current_section"]
    user_input = state.get("user_input", "")
    
    # Create section-specific prompts
    section_prompts = {
        "personal_info": """Extract personal information from the user's input. 
        Look for: name, date of birth, address. If information is missing, ask for it politely.
        Format the response to clearly indicate what information was collected and what's still needed.""",
        
        "contact_details": """Extract contact information from the user's input.
        Look for: email address, phone number. Validate email format and phone number format.
        If information is missing or invalid, ask for it politely.""",
        
        "employment_info": """Extract employment information from the user's input.
        Look for: job title, company name, years of experience. 
        If information is missing, ask for it politely.""",
        
        "preferences": """Extract preference information from the user's input.
        Look for: communication preferences (email/phone/text), interests, special requirements.
        If information is missing, ask for it politely."""
    }
    
    system_prompt = f"""You are processing the {current_section.replace('_', ' ')} section of a form.
    {section_prompts.get(current_section, '')}
    
    Current form data: {state['form_data']}
    User input: {user_input}
    
    Respond helpfully and ask for any missing required information for this section.
    If the section appears complete, confirm the information and indicate readiness to move to the next section."""
    
    messages = state["messages"] + [HumanMessage(content=user_input)] if user_input else state["messages"]
    messages.append(SystemMessage(content=system_prompt))
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response]
    }


def validate_section_node(state: FormState) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
    """Validate the current section data."""
    current_section = state["current_section"]
    user_input = state.get("user_input") or ""
    form_data = state["form_data"].copy()
    validation_errors: List[str] = []
    
    # Simple validation logic for each section
    if current_section == "personal_info":
        # Extract and validate personal info
        if "name" not in user_input.lower() and "name" not in form_data:
            validation_errors.append("Name is required for personal information section")
        else:
            # Simple name extraction (in real implementation, use more sophisticated NLP)
            if "name" in user_input.lower():
                form_data["name"] = user_input  # Simplified extraction
    
    elif current_section == "contact_details":
        # Validate email and phone
        if "@" in user_input:
            form_data["email"] = user_input.split()[-1] if "@" in user_input.split()[-1] else ""
        if any(char.isdigit() for char in user_input):
            # Simple phone extraction
            phone = ''.join(filter(str.isdigit, user_input))
            if len(phone) >= 10:
                form_data["phone"] = phone
    
    elif current_section == "employment_info":
        # Extract employment info
        if user_input:
            form_data[f"{current_section}_data"] = user_input
    
    elif current_section == "preferences":
        # Extract preferences
        if user_input:
            form_data[f"{current_section}_data"] = user_input
    
    return {
        "form_data": form_data,
        "validation_errors": validation_errors
    }


def complete_section_node(state: FormState) -> Dict[str, Union[str, List[str]]]:
    """Mark current section as complete and prepare for next section."""
    current_section = state["current_section"]
    completed_sections = state["completed_sections"].copy()
    available_sections = state["available_sections"]
    
    # Mark current section as completed
    if current_section not in completed_sections:
        completed_sections.append(current_section)
    
    # Determine next section
    current_index = available_sections.index(current_section)
    next_section = ""
    
    if current_index + 1 < len(available_sections):
        next_section = available_sections[current_index + 1]
    
    return {
        "completed_sections": completed_sections,
        "current_section": next_section
    }


def finalize_form_node(state: FormState) -> Dict[str, List[AnyMessage]]:
    """Finalize the form and provide summary."""
    llm = get_llm()
    
    form_summary = f"""
    Form completion summary:
    - Completed sections: {', '.join(state['completed_sections'])}
    - Form data collected: {state['form_data']}
    
    Thank you for completing the form! Here's a summary of the information collected.
    """
    
    system_message = SystemMessage(
        content=f"Provide a friendly summary of the completed form: {form_summary}"
    )
    
    response = llm.invoke([system_message])
    
    return {
        "messages": [response]
    }


# Conditional edge functions for routing
def route_after_processing(state: FormState) -> Literal["validate_section", "complete_section"]:
    """Route to validation or completion based on section status."""
    # Simple routing logic - in real implementation, use more sophisticated checks
    user_input = state.get("user_input", "")
    
    # If user provided substantial input, validate it
    if user_input and len(user_input.strip()) > 10:
        return "validate_section"
    else:
        return "complete_section"


def route_after_validation(state: FormState) -> Literal["process_section", "complete_section"]:
    """Route based on validation results."""
    validation_errors = state.get("validation_errors", [])
    
    # If there are validation errors, go back to processing
    if validation_errors:
        return "process_section"
    else:
        return "complete_section"


def route_after_completion(state: FormState) -> str:
    """Route to next section or finalize form."""
    current_section = state["current_section"]
    available_sections = state["available_sections"]
    
    # If current_section is empty, we've completed all sections
    if not current_section:
        return "finalize_form"
    
    # If we still have sections to process
    if current_section in available_sections:
        return "process_section"
    
    # Otherwise, finalize
    return "finalize_form"


def should_continue(state: FormState) -> str:
    """Determine if we should continue processing or end."""
    completed_sections = state["completed_sections"]
    available_sections = state["available_sections"]
    
    # If all sections are completed, end the process
    if len(completed_sections) >= len(available_sections):
        return END
    
    return "process_section"


# Build the StateGraph
def create_form_agent() -> StateGraph:
    """Create and return the compiled form filling agent graph."""
    
    # Initialize the StateGraph with our custom state
    graph_builder = StateGraph(FormState)
    
    # Add nodes
    graph_builder.add_node("start_form", start_form_node)
    graph_builder.add_node("process_section", process_section_node)
    graph_builder.add_node("validate_section", validate_section_node)
    graph_builder.add_node("complete_section", complete_section_node)
    graph_builder.add_node("finalize_form", finalize_form_node)
    
    # Add edges
    # Entry point
    graph_builder.add_edge(START, "start_form")
    graph_builder.add_edge("start_form", "process_section")
    
    # Conditional edges for section processing flow
    graph_builder.add_conditional_edges(
        "process_section",
        route_after_processing,
        {
            "validate_section": "validate_section",
            "complete_section": "complete_section"
        }
    )
    
    graph_builder.add_conditional_edges(
        "validate_section", 
        route_after_validation,
        {
            "process_section": "process_section",
            "complete_section": "complete_section"
        }
    )
    
    graph_builder.add_conditional_edges(
        "complete_section",
        route_after_completion,
        {
            "process_section": "process_section",
            "finalize_form": "finalize_form",
            END: END
        }
    )
    
    # Final edge
    graph_builder.add_edge("finalize_form", END)
    
    # Compile the graph
    return graph_builder.compile()


# Create and export the compiled graph as 'app' for deployment
app = create_form_agent()


# Optional: Function to run the agent (for testing)
def run_form_agent(user_input: str = "") -> None:
    """Run the form filling agent with optional user input."""
    initial_state = {
        "messages": [],
        "current_section": "",
        "form_data": {},
        "completed_sections": [],
        "available_sections": [],
        "user_input": user_input,
        "validation_errors": []
    }
    
    result = app.invoke(initial_state)
    
    # Print the final messages
    for message in result.get("messages", []):
        if hasattr(message, 'content'):
            print(f"Assistant: {message.content}")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Form Filling Agent initialized successfully!")
    print("The compiled graph is available as 'app' for deployment.")






