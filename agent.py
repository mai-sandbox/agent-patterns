"""
LangGraph Form Filling Agent

A section-by-section form filling agent that guides users through completing
a multi-section form with validation and review capabilities.
"""

from typing import Dict, List, Any, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


class FormState(TypedDict):
    """State schema for the form filling agent."""
    messages: Annotated[List, add_messages]
    form_data: Dict[str, Any]
    current_section: str
    completed_sections: List[str]
    is_complete: bool


# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


def personal_info_node(state: FormState) -> Dict[str, Any]:
    """Process personal information section of the form."""
    
    # Check if this section is already completed
    if "personal_info" in state.get("completed_sections", []):
        return {
            "current_section": "contact_details",
            "messages": [AIMessage(content="Personal information already completed. Moving to contact details.")]
        }
    
    # Get existing form data
    form_data = state.get("form_data", {})
    messages = state.get("messages", [])
    
    # Create prompt for personal information collection
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful form filling assistant. You are currently collecting personal information.

Current form data: {form_data}

Ask the user for the following personal information if not already provided:
- Full name (first and last name)
- Date of birth (MM/DD/YYYY format)
- Gender (optional)
- Nationality

Be conversational and ask for one piece of information at a time if multiple items are missing.
If all personal information is complete, confirm the details and indicate readiness to move to the next section.

Respond in a friendly, professional manner."""),
        ("human", "{user_input}")
    ])
    
    # Get the last user message
    user_input = ""
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_input = last_message.content
    
    # Generate response
    response = llm.invoke(prompt.format_messages(
        form_data=form_data,
        user_input=user_input
    ))
    
    # Extract personal info from the conversation
    personal_info = form_data.get("personal_info", {})
    
    # Simple extraction logic (in a real implementation, you might use more sophisticated NLP)
    if user_input and not personal_info.get("full_name"):
        # Look for name patterns in user input
        words = user_input.split()
        if len(words) >= 2 and any(word.istitle() for word in words[:2]):
            personal_info["full_name"] = " ".join(words[:2])
    
    # Check if personal info section is complete
    required_fields = ["full_name"]
    section_complete = all(personal_info.get(field) for field in required_fields)
    
    updated_form_data = {**form_data, "personal_info": personal_info}
    
    if section_complete:
        completed_sections = state.get("completed_sections", [])
        if "personal_info" not in completed_sections:
            completed_sections = completed_sections + ["personal_info"]
        
        return {
            "messages": [response],
            "form_data": updated_form_data,
            "current_section": "contact_details",
            "completed_sections": completed_sections
        }
    else:
        return {
            "messages": [response],
            "form_data": updated_form_data,
            "current_section": "personal_info"
        }


def contact_details_node(state: FormState) -> Dict[str, Any]:
    """Process contact details section of the form."""
    
    # Check if this section is already completed
    if "contact_details" in state.get("completed_sections", []):
        return {
            "current_section": "employment_info",
            "messages": [AIMessage(content="Contact details already completed. Moving to employment information.")]
        }
    
    form_data = state.get("form_data", {})
    messages = state.get("messages", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful form filling assistant. You are currently collecting contact details.

Current form data: {form_data}

Ask the user for the following contact information if not already provided:
- Email address
- Phone number
- Street address
- City, State, ZIP code

Be conversational and ask for one piece of information at a time if multiple items are missing.
If all contact details are complete, confirm the details and indicate readiness to move to the next section.

Respond in a friendly, professional manner."""),
        ("human", "{user_input}")
    ])
    
    user_input = ""
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_input = last_message.content
    
    response = llm.invoke(prompt.format_messages(
        form_data=form_data,
        user_input=user_input
    ))
    
    # Extract contact info
    contact_info = form_data.get("contact_details", {})
    
    # Simple extraction logic
    if user_input:
        if "@" in user_input and not contact_info.get("email"):
            # Extract email
            words = user_input.split()
            for word in words:
                if "@" in word:
                    contact_info["email"] = word
                    break
        
        # Look for phone number patterns
        import re
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_match = re.search(phone_pattern, user_input)
        if phone_match and not contact_info.get("phone"):
            contact_info["phone"] = phone_match.group()
    
    # Check if contact details section is complete
    required_fields = ["email"]
    section_complete = all(contact_info.get(field) for field in required_fields)
    
    updated_form_data = {**form_data, "contact_details": contact_info}
    
    if section_complete:
        completed_sections = state.get("completed_sections", [])
        if "contact_details" not in completed_sections:
            completed_sections = completed_sections + ["contact_details"]
        
        return {
            "messages": [response],
            "form_data": updated_form_data,
            "current_section": "employment_info",
            "completed_sections": completed_sections
        }
    else:
        return {
            "messages": [response],
            "form_data": updated_form_data,
            "current_section": "contact_details"
        }


def employment_info_node(state: FormState) -> Dict[str, Any]:
    """Process employment information section of the form."""
    
    # Check if this section is already completed
    if "employment_info" in state.get("completed_sections", []):
        return {
            "current_section": "review_and_submit",
            "messages": [AIMessage(content="Employment information already completed. Moving to review and submit.")]
        }
    
    form_data = state.get("form_data", {})
    messages = state.get("messages", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful form filling assistant. You are currently collecting employment information.

Current form data: {form_data}

Ask the user for the following employment information if not already provided:
- Current job title
- Company name
- Years of experience
- Annual salary (optional)

Be conversational and ask for one piece of information at a time if multiple items are missing.
If all employment information is complete, confirm the details and indicate readiness to move to review.

Respond in a friendly, professional manner."""),
        ("human", "{user_input}")
    ])
    
    user_input = ""
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_input = last_message.content
    
    response = llm.invoke(prompt.format_messages(
        form_data=form_data,
        user_input=user_input
    ))
    
    # Extract employment info
    employment_info = form_data.get("employment_info", {})
    
    # Simple extraction logic for job title and company
    if user_input and not employment_info.get("job_title"):
        # This is a simplified approach - in practice, you'd use more sophisticated NLP
        if any(word in user_input.lower() for word in ["developer", "engineer", "manager", "analyst", "designer"]):
            employment_info["job_title"] = user_input.strip()
    
    # Check if employment info section is complete
    required_fields = ["job_title"]
    section_complete = all(employment_info.get(field) for field in required_fields)
    
    updated_form_data = {**form_data, "employment_info": employment_info}
    
    if section_complete:
        completed_sections = state.get("completed_sections", [])
        if "employment_info" not in completed_sections:
            completed_sections = completed_sections + ["employment_info"]
        
        return {
            "messages": [response],
            "form_data": updated_form_data,
            "current_section": "review_and_submit",
            "completed_sections": completed_sections
        }
    else:
        return {
            "messages": [response],
            "form_data": updated_form_data,
            "current_section": "employment_info"
        }


def review_and_submit_node(state: FormState) -> Dict[str, Any]:
    """Review all collected information and submit the form."""
    
    form_data = state.get("form_data", {})
    
    # Create a comprehensive review of all collected data
    review_content = "## Form Review\n\nPlease review all the information collected:\n\n"
    
    # Personal Information
    personal_info = form_data.get("personal_info", {})
    if personal_info:
        review_content += "**Personal Information:**\n"
        for key, value in personal_info.items():
            review_content += f"- {key.replace('_', ' ').title()}: {value}\n"
        review_content += "\n"
    
    # Contact Details
    contact_details = form_data.get("contact_details", {})
    if contact_details:
        review_content += "**Contact Details:**\n"
        for key, value in contact_details.items():
            review_content += f"- {key.replace('_', ' ').title()}: {value}\n"
        review_content += "\n"
    
    # Employment Information
    employment_info = form_data.get("employment_info", {})
    if employment_info:
        review_content += "**Employment Information:**\n"
        for key, value in employment_info.items():
            review_content += f"- {key.replace('_', ' ').title()}: {value}\n"
        review_content += "\n"
    
    review_content += "If everything looks correct, please confirm to submit the form. If you need to make changes, let me know which section you'd like to update."
    
    return {
        "messages": [AIMessage(content=review_content)],
        "current_section": "review_and_submit",
        "is_complete": True
    }


def route_sections(state: FormState) -> str:
    """Route to the appropriate section based on current state."""
    current_section = state.get("current_section", "personal_info")
    
    if current_section == "personal_info":
        return "personal_info"
    elif current_section == "contact_details":
        return "contact_details"
    elif current_section == "employment_info":
        return "employment_info"
    elif current_section == "review_and_submit":
        return "review_and_submit"
    else:
        return "personal_info"  # Default to start


def should_continue(state: FormState) -> str:
    """Determine if the form process should continue or end."""
    if state.get("is_complete", False):
        return "end"
    else:
        return "continue"


# Create the StateGraph
workflow = StateGraph(FormState)

# Add nodes
workflow.add_node("personal_info", personal_info_node)
workflow.add_node("contact_details", contact_details_node)
workflow.add_node("employment_info", employment_info_node)
workflow.add_node("review_and_submit", review_and_submit_node)

# Set entry point
workflow.set_entry_point("personal_info")

# Add conditional edges for section routing
workflow.add_conditional_edges(
    "personal_info",
    route_sections,
    {
        "personal_info": "personal_info",
        "contact_details": "contact_details",
        "employment_info": "employment_info",
        "review_and_submit": "review_and_submit"
    }
)

workflow.add_conditional_edges(
    "contact_details",
    route_sections,
    {
        "personal_info": "personal_info",
        "contact_details": "contact_details",
        "employment_info": "employment_info",
        "review_and_submit": "review_and_submit"
    }
)

workflow.add_conditional_edges(
    "employment_info",
    route_sections,
    {
        "personal_info": "personal_info",
        "contact_details": "contact_details",
        "employment_info": "employment_info",
        "review_and_submit": "review_and_submit"
    }
)

workflow.add_conditional_edges(
    "review_and_submit",
    should_continue,
    {
        "continue": "personal_info",
        "end": END
    }
)

# Compile the graph with memory for persistence
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)
