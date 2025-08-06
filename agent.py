"""
LangGraph Form Filling Agent

A section-by-section form filling agent with human-in-the-loop validation.
Processes forms by collecting information for each section and validating
with human reviewers before proceeding to the next section.
"""

from typing import Annotated, Dict, Any, List, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


# Custom State class with form-specific fields
class FormState(TypedDict):
    """State for the form filling agent with section-specific fields."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_section: str
    personal_info: Dict[str, Any]
    contact_info: Dict[str, Any]
    employment_info: Dict[str, Any]
    additional_info: Dict[str, Any]
    form_complete: bool
    validation_status: str


# Form sections configuration
FORM_SECTIONS = {
    "personal": {
        "name": "Personal Information",
        "fields": ["full_name", "date_of_birth", "gender", "nationality"],
        "description": "Basic personal details including name, birth date, and nationality"
    },
    "contact": {
        "name": "Contact Information", 
        "fields": ["email", "phone", "address", "city", "postal_code"],
        "description": "Contact details including email, phone, and address"
    },
    "employment": {
        "name": "Employment Information",
        "fields": ["job_title", "company", "years_experience", "salary_range"],
        "description": "Professional background and employment details"
    },
    "additional": {
        "name": "Additional Information",
        "fields": ["skills", "certifications", "references", "comments"],
        "description": "Additional qualifications and supplementary information"
    }
}


# Tools for form processing
@tool
def collect_section_data(
    section: str,
    data: Dict[str, Any],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Collect and validate data for a specific form section."""
    
    if section not in FORM_SECTIONS:
        return f"Error: Unknown section '{section}'"
    
    section_info = FORM_SECTIONS[section]
    
    # Validate that all required fields are present
    missing_fields = []
    for field in section_info["fields"]:
        if field not in data or not data[field]:
            missing_fields.append(field)
    
    if missing_fields:
        return f"Missing required fields for {section_info['name']}: {', '.join(missing_fields)}"
    
    # Request human validation for this section
    human_response = interrupt({
        "question": f"Please review the {section_info['name']} data:",
        "section": section,
        "data": data,
        "fields": section_info["fields"],
        "instructions": "Please verify the information is correct. Respond with 'approved' to continue or provide corrections."
    })
    
    # Process human response
    if human_response.get("status") == "approved":
        response = f"✅ {section_info['name']} approved and saved"
        # Update the appropriate section in state
        state_update = {
            f"{section}_info": data,
            "validation_status": "approved",
            "messages": [ToolMessage(response, tool_call_id=tool_call_id)]
        }
    else:
        # Handle corrections from human reviewer
        corrected_data = human_response.get("corrected_data", data)
        response = f"📝 {section_info['name']} updated with corrections"
        state_update = {
            f"{section}_info": corrected_data,
            "validation_status": "corrected",
            "messages": [ToolMessage(response, tool_call_id=tool_call_id)]
        }
    
    return Command(update=state_update)


@tool
def get_next_section(current_section: str) -> str:
    """Determine the next section to process in the form."""
    sections = list(FORM_SECTIONS.keys())
    
    if not current_section:
        return sections[0]  # Start with first section
    
    try:
        current_index = sections.index(current_section)
        if current_index < len(sections) - 1:
            return sections[current_index + 1]
        else:
            return "complete"  # All sections done
    except ValueError:
        return sections[0]  # Default to first section if current not found


@tool
def validate_form_completion(
    personal_info: Dict[str, Any],
    contact_info: Dict[str, Any], 
    employment_info: Dict[str, Any],
    additional_info: Dict[str, Any]
) -> str:
    """Validate that all form sections are complete."""
    
    sections_status = {}
    
    # Check each section for completeness
    for section_key, section_config in FORM_SECTIONS.items():
        section_data = locals().get(f"{section_key}_info", {})
        required_fields = section_config["fields"]
        
        missing_fields = [field for field in required_fields if not section_data.get(field)]
        sections_status[section_key] = {
            "complete": len(missing_fields) == 0,
            "missing_fields": missing_fields
        }
    
    # Check if all sections are complete
    all_complete = all(status["complete"] for status in sections_status.values())
    
    if all_complete:
        return "✅ Form is complete and ready for final submission"
    else:
        incomplete_sections = [
            f"{FORM_SECTIONS[key]['name']}: missing {', '.join(status['missing_fields'])}"
            for key, status in sections_status.items()
            if not status["complete"]
        ]
        return f"❌ Form incomplete. Missing data in: {'; '.join(incomplete_sections)}"


# Initialize the LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.1
)

# Bind tools to the LLM
tools = [collect_section_data, get_next_section, validate_form_completion]
llm_with_tools = llm.bind_tools(tools)


# Node functions
def form_coordinator(state: FormState) -> Dict[str, Any]:
    """Main coordinator node that manages the form filling process."""
    
    messages = state["messages"]
    current_section = state.get("current_section", "")
    
    # Create system message with current context
    system_prompt = f"""You are a helpful form filling assistant. Your job is to guide users through completing a form section by section.

Current section: {current_section or 'Starting'}
Available sections: {', '.join(FORM_SECTIONS.keys())}

For each section, you should:
1. Ask the user for the required information
2. Use the collect_section_data tool to validate and save the data
3. Move to the next section using get_next_section tool
4. Use validate_form_completion when all sections are done

Be conversational and helpful. Explain what information is needed for each section.
"""
    
    # Add system context to messages
    context_messages = [HumanMessage(content=system_prompt)] + messages
    
    # Get response from LLM
    response = llm_with_tools.invoke(context_messages)
    
    return {"messages": [response]}


def section_processor(state: FormState) -> Dict[str, Any]:
    """Process the current section and determine next steps."""
    
    current_section = state.get("current_section", "")
    
    # If no current section, start with the first one
    if not current_section:
        next_section = list(FORM_SECTIONS.keys())[0]
        return {
            "current_section": next_section,
            "messages": [AIMessage(content=f"Let's start with the {FORM_SECTIONS[next_section]['name']} section.")]
        }
    
    # Check if current section is complete
    section_data = state.get(f"{current_section}_info", {})
    section_fields = FORM_SECTIONS[current_section]["fields"]
    
    # If section is complete, move to next
    if all(section_data.get(field) for field in section_fields):
        sections = list(FORM_SECTIONS.keys())
        try:
            current_index = sections.index(current_section)
            if current_index < len(sections) - 1:
                next_section = sections[current_index + 1]
                return {
                    "current_section": next_section,
                    "messages": [AIMessage(content=f"Great! Now let's move to the {FORM_SECTIONS[next_section]['name']} section.")]
                }
            else:
                return {
                    "form_complete": True,
                    "messages": [AIMessage(content="🎉 All sections completed! Let me validate the complete form.")]
                }
        except ValueError:
            pass
    
    return {"messages": []}


def completion_checker(state: FormState) -> Dict[str, Any]:
    """Check if the form is complete and ready for submission."""
    
    # Check all sections
    all_sections_complete = True
    for section_key in FORM_SECTIONS.keys():
        section_data = state.get(f"{section_key}_info", {})
        section_fields = FORM_SECTIONS[section_key]["fields"]
        
        if not all(section_data.get(field) for field in section_fields):
            all_sections_complete = False
            break
    
    if all_sections_complete:
        return {
            "form_complete": True,
            "messages": [AIMessage(content="✅ Form validation complete! All sections have been filled out and approved.")]
        }
    else:
        return {
            "form_complete": False,
            "messages": [AIMessage(content="❌ Form is not yet complete. Please continue filling out the remaining sections.")]
        }


# Conditional edge functions
def should_continue_processing(state: FormState) -> Literal["continue", "complete", "validate"]:
    """Determine the next step in the form processing workflow."""
    
    # Check if form is marked as complete
    if state.get("form_complete", False):
        return "complete"
    
    # Check if we have tool calls to process
    last_message = state["messages"][-1] if state["messages"] else None
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    
    # Check if all sections have data (even if not validated)
    sections_with_data = 0
    for section_key in FORM_SECTIONS.keys():
        section_data = state.get(f"{section_key}_info", {})
        if section_data:
            sections_with_data += 1
    
    if sections_with_data == len(FORM_SECTIONS):
        return "validate"
    
    return "continue"


# Build the StateGraph
def create_form_filling_graph():
    """Create and compile the form filling graph."""
    
    # Initialize the graph
    graph_builder = StateGraph(FormState)
    
    # Add nodes
    graph_builder.add_node("coordinator", form_coordinator)
    graph_builder.add_node("section_processor", section_processor)
    graph_builder.add_node("completion_checker", completion_checker)
    
    # Add edges
    graph_builder.add_edge(START, "coordinator")
    
    # Add conditional edges
    graph_builder.add_conditional_edges(
        "coordinator",
        should_continue_processing,
        {
            "continue": "section_processor",
            "validate": "completion_checker", 
            "complete": END
        }
    )
    
    graph_builder.add_edge("section_processor", "coordinator")
    graph_builder.add_edge("completion_checker", END)
    
    # Compile with memory for persistence
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


# Create and export the compiled graph
graph = create_form_filling_graph()

# MANDATORY: Export as 'app' for deployment
app = graph
