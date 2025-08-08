"""LangGraph Support Ticket Triage Agent.

This module implements a complete support ticket triage system using LangGraph
that classifies, prioritizes, summarizes, routes, and drafts acknowledgements
for customer support tickets.
"""

import os
from typing import Dict, Any, cast
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from state import TicketState

# Load environment variables
load_dotenv()


class SupportTicketTriageAgent:
    """LangGraph-based Support Ticket Triage Agent.
    
    This agent processes customer support tickets through a 5-step workflow:
    1. Classify ticket category
    2. Detect priority level
    3. Summarize ticket content
    4. Route to appropriate email
    5. Draft acknowledgement message
    """
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """Initialize the Support Ticket Triage Agent.
        
        Args:
            model_name: The Anthropic model to use for processing
            
        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.llm = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=0.1
        )
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow for ticket triage.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Create the state graph
        workflow = StateGraph(TicketState)
        
        # Add nodes for each processing step
        workflow.add_node("classify_ticket", self.classify_ticket)
        workflow.add_node("detect_priority", self.detect_priority)
        workflow.add_node("summarize_ticket", self.summarize_ticket)
        workflow.add_node("route_email", self.route_email)
        workflow.add_node("draft_acknowledgement", self.draft_acknowledgement)
        
        # Define the sequential flow
        workflow.set_entry_point("classify_ticket")
        workflow.add_edge("classify_ticket", "detect_priority")
        workflow.add_edge("detect_priority", "summarize_ticket")
        workflow.add_edge("summarize_ticket", "route_email")
        workflow.add_edge("route_email", "draft_acknowledgement")
        workflow.add_edge("draft_acknowledgement", END)
        
        return workflow.compile()
    
    def classify_ticket(self, state: TicketState) -> Dict[str, Any]:
        """Classify the ticket into one of three categories.
        
        Args:
            state: Current ticket state containing ticket_text
            
        Returns:
            Updated state with category field set
        """
        try:
            prompt = f"""
            Classify the following customer support ticket into exactly one of these categories:
            - "Billing" (payment issues, invoices, subscription problems, refunds)
            - "Technical" (software bugs, login issues, feature problems, system errors)
            - "General Inquiry" (questions, information requests, general support)
            
            Ticket: {state['ticket_text']}
            
            Respond with only the category name: Billing, Technical, or General Inquiry
            """
            
            response = self.llm.invoke(prompt)
            category = str(response.content).strip()
            
            # Validate category
            valid_categories = ["Billing", "Technical", "General Inquiry"]
            if category not in valid_categories:
                # Default to General Inquiry if classification is unclear
                category = "General Inquiry"
            
            return {"category": category}
            
        except Exception:
            # Error handling - default to General Inquiry
            return {"category": "General Inquiry"}
    
    def detect_priority(self, state: TicketState) -> Dict[str, Any]:
        """Detect the priority level of the ticket.
        
        Args:
            state: Current ticket state with ticket_text and category
            
        Returns:
            Updated state with priority field set
        """
        try:
            prompt = f"""
            Analyze the urgency of this {state['category']} support ticket and classify it as:
            - "High" (urgent issues, system down, payment failures, security concerns, angry customers)
            - "Medium" (important but not urgent, feature requests, moderate issues)
            - "Low" (general questions, minor issues, information requests)
            
            Ticket: {state['ticket_text']}
            
            Look for urgency indicators like:
            - Words: urgent, emergency, critical, immediately, ASAP, broken, down, not working
            - Customer tone: frustrated, angry, demanding immediate action
            - Business impact: affecting multiple users, preventing work, financial impact
            
            Respond with only the priority level: High, Medium, or Low
            """
            
            response = self.llm.invoke(prompt)
            priority = str(response.content).strip()
            
            # Validate priority
            valid_priorities = ["High", "Medium", "Low"]
            if priority not in valid_priorities:
                # Default to Medium if priority detection is unclear
                priority = "Medium"
            
            return {"priority": priority}
            
        except Exception:
            # Error handling - default to Medium priority
            return {"priority": "Medium"}
    
    def summarize_ticket(self, state: TicketState) -> Dict[str, Any]:
        """Create a one-sentence summary of the ticket.
        
        Args:
            state: Current ticket state with ticket_text, category, and priority
            
        Returns:
            Updated state with summary field set
        """
        try:
            prompt = f"""
            Create a clear, concise one-sentence summary of this {state['category']} support ticket.
            The summary should capture the main issue or request in professional language.
            
            Ticket: {state['ticket_text']}
            
            Provide only the summary sentence, no additional text.
            """
            
            response = self.llm.invoke(prompt)
            summary = str(response.content).strip()
            
            # Ensure it's a single sentence
            if not summary.endswith('.'):
                summary += '.'
            
            return {"summary": summary}
            
        except Exception:
            # Error handling - create basic summary
            category = state.get('category', 'support')
            return {"summary": f"Customer submitted a {category.lower()} request requiring assistance."}
    
    def route_email(self, state: TicketState) -> Dict[str, Any]:
        """Route the ticket to the appropriate email based on category and priority.
        
        Args:
            state: Current ticket state with category and priority
            
        Returns:
            Updated state with email field set
        """
        try:
            category = state.get('category', 'General Inquiry')
            priority = state.get('priority', 'Medium')
            
            # Apply routing logic
            if category == "Billing":
                if priority == "High":
                    email = "priority-billing@company.com"
                else:
                    email = "billing@company.com"
            elif category == "Technical":
                if priority == "High":
                    email = "urgent-tech@company.com"
                else:
                    email = "tech@company.com"
            else:  # General Inquiry
                email = "support@company.com"
            
            return {"email": email}
            
        except Exception:
            # Error handling - default to general support
            return {"email": "support@company.com"}
    
    def draft_acknowledgement(self, state: TicketState) -> Dict[str, Any]:
        """Draft an acknowledgement email snippet.
        
        Args:
            state: Current ticket state with summary and email
            
        Returns:
            Updated state with acknowledgement field set
        """
        try:
            summary = state.get('summary', 'your request')
            email = state.get('email', 'our support team')
            category = state.get('category', 'support')
            priority = state.get('priority', 'Medium')
            
            prompt = f"""
            Draft a professional 1-2 sentence acknowledgement email snippet for a customer who submitted:
            Summary: {summary}
            Category: {category}
            Priority: {priority}
            Routed to: {email}
            
            The acknowledgement should:
            - Thank the customer
            - Reference their specific issue briefly
            - Indicate it's been routed appropriately
            - Be professional and reassuring
            
            Provide only the acknowledgement text, no subject line or signatures.
            """
            
            response = self.llm.invoke(prompt)
            acknowledgement = str(response.content).strip()
            
            return {"acknowledgement": acknowledgement}
            
        except Exception:
            # Error handling - create basic acknowledgement
            return {"acknowledgement": "Thank you for contacting us. Your request has been received and routed to the appropriate team for assistance."}
    
    def process_ticket(self, ticket_text: str) -> TicketState:
        """Process a support ticket through the complete triage workflow.
        
        Args:
            ticket_text: Raw customer support ticket text
            
        Returns:
            Complete TicketState with all fields populated
            
        Raises:
            ValueError: If ticket_text is empty or None
        """
        if not ticket_text or not ticket_text.strip():
            raise ValueError("Ticket text cannot be empty")
        
        # Initialize state
        initial_state: TicketState = {
            "ticket_text": ticket_text.strip(),
            "category": "",
            "priority": "",
            "summary": "",
            "email": "",
            "acknowledgement": ""
        }
        
        # Execute the workflow
        result = self.graph.invoke(initial_state)
        
        # Type cast to ensure proper return type
        return cast(TicketState, result)


def main():
    """Example usage of the Support Ticket Triage Agent."""
    # Example ticket for demonstration
    sample_ticket = """
    Hi, I'm having trouble logging into my account. I've tried resetting my password 
    multiple times but I keep getting an error message saying 'Invalid credentials'. 
    This is really urgent because I need to access my project files for a client 
    presentation tomorrow morning. Can someone please help me ASAP?
    """
    
    try:
        # Initialize the agent
        agent = SupportTicketTriageAgent()
        
        # Process the ticket
        result = agent.process_ticket(sample_ticket)
        
        # Display results
        print("=== Support Ticket Triage Results ===")
        print(f"Original Ticket: {result['ticket_text'][:100]}...")
        print(f"Category: {result['category']}")
        print(f"Priority: {result['priority']}")
        print(f"Summary: {result['summary']}")
        print(f"Routed to: {result['email']}")
        print(f"Acknowledgement: {result['acknowledgement']}")
        
    except Exception as e:
        print(f"Error processing ticket: {e}")


if __name__ == "__main__":
    main()
















