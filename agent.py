"""
LangGraph Form Filling Agent

This agent processes forms section by section, guiding users through:
1. Personal Information
2. Contact Details  
3. Preferences
4. Review and Submit

The agent maintains state across sections and provides a conversational interface
for form completion.
"""

from typing import Annotated, Dict, List, Literal, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class FormState(TypedDict):
    """State for the form filling agent."""
    messages: Annotated[List[Any], add_messages]
    current_section: str
    form_data: Dict[str, Any]
    completed_sections: List[str]


# Initialize the language model
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")


def personal_info_node(state: FormState) -> Dict[str, Any]:
    """Handle personal information collection."""
    messages = state["messages"]
    form_data = state.get("form_data", {})
    
    # System prompt for personal info collection
    system_prompt = """You are a helpful form filling assistant. You are currently collecting PERSONAL INFORMATION.

Ask for the following information if not already provided:
- Full name
- Date of birth
- Gender (optional)
- Nationality

Be conversational and friendly. If the user provides information, acknowledge it and ask for any missing details.
If all personal information is complete, let the user know this section is done and you'll move to the next section.

Current form data: {form_data}
""".format(form_data=form_data.get("personal_info", {}))

    # Create messages for LLM
    llm_messages = [SystemMessage(content=system_prompt)] + messages
    
    # Get LLM response
    response = llm.invoke(llm_messages)
    
    # Check if personal info section is complete
    personal_info = form_data.get("personal_info", {})
    is_complete = bool(
        personal_info.get("full_name") and 
        personal_info.get("date_of_birth")
    )
    
    # Update state
    updates = {
        "messages": [response],
        "current_section": "personal_info"
    }
    
    if is_complete and "personal_info" not in state.get("completed_sections", []):
        completed = state.get("completed_sections", []).copy()
        completed.append("personal_info")
        updates["completed_sections"] = completed
    
    return updates


def contact_details_node(state: FormState) -> Dict[str, Any]:
    """Handle contact details collection."""
    messages = state["messages"]
    form_data = state.get("form_data", {})
    
    system_prompt = """You are a helpful form filling assistant. You are currently collecting CONTACT DETAILS.

Ask for the following information if not already provided:
- Email address
- Phone number
- Address (street, city, state/province, postal code, country)

Be conversational and friendly. If the user provides information, acknowledge it and ask for any missing details.
If all contact details are complete, let the user know this section is done and you'll move to the next section.

Current form data: {form_data}
""".format(form_data=form_data.get("contact_details", {}))

    llm_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(llm_messages)
    
    # Check if contact details section is complete
    contact_details = form_data.get("contact_details", {})
    is_complete = bool(
        contact_details.get("email") and 
        contact_details.get("phone") and
        contact_details.get("address")
    )
    
    updates = {
        "messages": [response],
        "current_section": "contact_details"
    }
    
    if is_complete and "contact_details" not in state.get("completed_sections", []):
        completed = state.get("completed_sections", []).copy()
        completed.append("contact_details")
        updates["completed_sections"] = completed
    
    return updates


def preferences_node(state: FormState) -> Dict[str, Any]:
    """Handle preferences collection."""
    messages = state["messages"]
    form_data = state.get("form_data", {})
    
    system_prompt = """You are a helpful form filling assistant. You are currently collecting PREFERENCES.

Ask for the following information if not already provided:
- Communication preferences (email, phone, mail)
- Language preference
- Any special requirements or notes

Be conversational and friendly. If the user provides information, acknowledge it and ask for any missing details.
If all preferences are complete, let the user know this section is done and you'll move to review.

Current form data: {form_data}
""".format(form_data=form_data.get("preferences", {}))

    llm_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(llm_messages)
    
    # Check if preferences section is complete
    preferences = form_data.get("preferences", {})
    is_complete = bool(
        preferences.get("communication_preference") and 
        preferences.get("language_preference")
    )
    
    updates = {
        "messages": [response],
        "current_section": "preferences"
    }
    
    if is_complete and "preferences" not in state.get("completed_sections", []):
        completed = state.get("completed_sections", []).copy()
        completed.append("preferences")
        updates["completed_sections"] = completed
    
    return updates


def review_submit_node(state: FormState) -> Dict[str, Any]:
    """Handle form review and submission."""
    messages = state["messages"]
    form_data = state.get("form_data", {})
    
    system_prompt = """You are a helpful form filling assistant. You are now in the REVIEW AND SUBMIT section.

Present a summary of all the information collected:
- Personal Information: {personal_info}
- Contact Details: {contact_details}  
- Preferences: {preferences}

Ask the user to review the information and confirm if they want to submit the form.
If they want to make changes, guide them on what they can modify.
If they confirm submission, congratulate them on completing the form.

""".format(
        personal_info=form_data.get("personal_info", {}),
        contact_details=form_data.get("contact_details", {}),
        preferences=form_data.get("preferences", {})
    )

    llm_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(llm_messages)
    
    updates = {
        "messages": [response],
        "current_section": "review_submit"
    }
    
    return updates


def should_continue(state: FormState) -> Literal["personal_info", "contact_details", "preferences", "review_submit", "end"]:
    """Determine which section to go to next based on completion status."""
    completed_sections = state.get("completed_sections", [])
    current_section = state.get("current_section", "")
    
    # If we're in review and user confirms submission, end
    if current_section == "review_submit":
        # Check if user confirmed submission in the last message
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            if any(word in content for word in ["submit", "confirm", "yes", "done", "finish"]):
                return "end"
    
    # Route to next incomplete section
    if "personal_info" not in completed_sections:
        return "personal_info"
    elif "contact_details" not in completed_sections:
        return "contact_details"
    elif "preferences" not in completed_sections:
        return "preferences"
    else:
        return "review_submit"


def initialize_form_state(state: FormState) -> Dict[str, Any]:
    """Initialize the form with welcome message and empty form data."""
    welcome_message = AIMessage(content="""
Welcome to the Form Filling Assistant! 🌟

I'll help you complete your form step by step. We'll go through these sections:
1. Personal Information
2. Contact Details
3. Preferences
4. Review and Submit

Let's start with your personal information. What's your full name?
""")
    
    return {
        "messages": [welcome_message],
        "current_section": "personal_info",
        "form_data": {
            "personal_info": {},
            "contact_details": {},
            "preferences": {}
        },
        "completed_sections": []
    }


# Create the StateGraph
graph_builder = StateGraph(FormState)

# Add nodes
graph_builder.add_node("initialize", initialize_form_state)
graph_builder.add_node("personal_info", personal_info_node)
graph_builder.add_node("contact_details", contact_details_node)
graph_builder.add_node("preferences", preferences_node)
graph_builder.add_node("review_submit", review_submit_node)

# Add edges
graph_builder.add_edge(START, "initialize")
graph_builder.add_edge("initialize", "personal_info")

# Add conditional edges for section routing
graph_builder.add_conditional_edges(
    "personal_info",
    should_continue,
    {
        "personal_info": "personal_info",
        "contact_details": "contact_details",
        "preferences": "preferences",
        "review_submit": "review_submit",
        "end": END
    }
)

graph_builder.add_conditional_edges(
    "contact_details",
    should_continue,
    {
        "personal_info": "personal_info",
        "contact_details": "contact_details", 
        "preferences": "preferences",
        "review_submit": "review_submit",
        "end": END
    }
)

graph_builder.add_conditional_edges(
    "preferences",
    should_continue,
    {
        "personal_info": "personal_info",
        "contact_details": "contact_details",
        "preferences": "preferences", 
        "review_submit": "review_submit",
        "end": END
    }
)

graph_builder.add_conditional_edges(
    "review_submit",
    should_continue,
    {
        "personal_info": "personal_info",
        "contact_details": "contact_details",
        "preferences": "preferences",
        "review_submit": "review_submit",
        "end": END
    }
)

# Compile the graph
graph = graph_builder.compile()

# Export as 'app' for deployment compatibility
app = graph
