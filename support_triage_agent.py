"""
LangGraph Support Ticket Triage Agent

This module implements a comprehensive support ticket triage system using LangGraph.
The agent processes customer support tickets through a 5-step workflow:
1. Classification (Billing/Technical/General Inquiry)
2. Priority Detection (Low/Medium/High)
3. Summarization (one-sentence summary)
4. Routing (email assignment based on category + priority)
5. Acknowledgment (draft response snippet)
"""

import logging
import os
from typing import Dict, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TicketState(TypedDict):
    """State schema for the support ticket triage workflow."""
    ticket_text: str
    category: str
    priority: str
    summary: str
    routing_email: str
    acknowledgment: str


class SupportTriageAgent:
    """
    LangGraph-based support ticket triage agent that processes customer tickets
    through a structured 5-step workflow.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Support Triage Agent.
        
        Args:
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            api_key=self.api_key,
            temperature=0.1
        )
        
        # Email routing rules
        self.routing_rules = {
            ("Billing", "High"): "priority-billing@company.com",
            ("Billing", "Medium"): "billing@company.com",
            ("Billing", "Low"): "billing@company.com",
            ("Technical", "High"): "urgent-tech@company.com",
            ("Technical", "Medium"): "tech@company.com",
            ("Technical", "Low"): "tech@company.com",
            ("General Inquiry", "High"): "support@company.com",
            ("General Inquiry", "Medium"): "support@company.com",
            ("General Inquiry", "Low"): "support@company.com",
        }
        
        # Build the graph
        self.graph = self._build_graph()
        logger.info("Support Triage Agent initialized successfully")
    
    def classify_ticket(self, state: TicketState) -> Dict[str, Any]:
        """
        Node 1: Classify the ticket into one of three categories.
        
        Args:
            state: Current ticket state
            
        Returns:
            Updated state with category field
        """
        try:
            logger.info("Starting ticket classification")
            
            system_prompt = """You are a support ticket classifier. Your job is to categorize customer support tickets into exactly one of these three categories:

1. "Billing" - Issues related to payments, invoices, subscriptions, refunds, pricing, account charges
2. "Technical" - Technical problems, bugs, feature requests, integration issues, performance problems
3. "General Inquiry" - General questions, information requests, feedback, complaints not related to billing or technical issues

Respond with ONLY the category name: "Billing", "Technical", or "General Inquiry"."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Classify this support ticket:\n\n{state['ticket_text']}")
            ]
            
            response = self.llm.invoke(messages)
            category = str(response.content).strip()
            
            # Validate category
            valid_categories = ["Billing", "Technical", "General Inquiry"]
            if category not in valid_categories:
                logger.warning(f"Invalid category '{category}', defaulting to 'General Inquiry'")
                category = "General Inquiry"
            
            logger.info(f"Ticket classified as: {category}")
            return {"category": category}
            
        except Exception as e:
            logger.error(f"Error in classify_ticket: {str(e)}")
            return {"category": "General Inquiry"}  # Default fallback
    
    def detect_priority(self, state: TicketState) -> Dict[str, Any]:
        """
        Node 2: Detect the priority level based on urgency clues.
        
        Args:
            state: Current ticket state
            
        Returns:
            Updated state with priority field
        """
        try:
            logger.info("Starting priority detection")
            
            system_prompt = """You are a support ticket priority detector. Analyze the urgency and impact of the ticket to determine priority level.

Priority Guidelines:
- "High": Critical issues affecting business operations, security breaches, complete service outages, urgent deadlines mentioned, angry/frustrated tone with immediate needs
- "Medium": Important issues with moderate impact, partial functionality problems, requests with reasonable timelines
- "Low": Minor issues, general questions, feature requests, cosmetic problems, no urgency indicated

Look for urgency indicators like:
- Words: urgent, critical, emergency, ASAP, immediately, broken, down, not working
- Business impact: can't work, blocking, production issue, losing money
- Emotional tone: frustrated, angry, desperate

Respond with ONLY the priority level: "High", "Medium", or "Low"."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Determine the priority for this {state['category']} ticket:\n\n{state['ticket_text']}")
            ]
            
            response = self.llm.invoke(messages)
            priority = str(response.content).strip()
            
            # Validate priority
            valid_priorities = ["High", "Medium", "Low"]
            if priority not in valid_priorities:
                logger.warning(f"Invalid priority '{priority}', defaulting to 'Medium'")
                priority = "Medium"
            
            logger.info(f"Priority detected as: {priority}")
            return {"priority": priority}
            
        except Exception as e:
            logger.error(f"Error in detect_priority: {str(e)}")
            return {"priority": "Medium"}  # Default fallback
    
    def summarize_ticket(self, state: TicketState) -> Dict[str, Any]:
        """
        Node 3: Generate a one-sentence summary of the ticket.
        
        Args:
            state: Current ticket state
            
        Returns:
            Updated state with summary field
        """
        try:
            logger.info("Starting ticket summarization")
            
            system_prompt = """You are a support ticket summarizer. Create a clear, concise one-sentence summary that captures the main issue or request.

Guidelines:
- Keep it to exactly one sentence
- Include the key problem or request
- Be specific but concise
- Use professional language
- Focus on what the customer needs or what's wrong

Examples:
- "Customer is unable to log into their account after password reset"
- "User requests refund for duplicate billing charge from last month"
- "Customer experiencing slow page load times on the dashboard"
- "User asking about integration options with third-party CRM systems"

Respond with ONLY the one-sentence summary."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Summarize this {state['category']} ticket in one sentence:\n\n{state['ticket_text']}")
            ]
            
            response = self.llm.invoke(messages)
            summary = str(response.content).strip()
            
            # Ensure it's a single sentence (basic validation)
            if not summary.endswith('.'):
                summary += '.'
            
            logger.info(f"Ticket summarized: {summary[:100]}...")
            return {"summary": summary}
            
        except Exception as e:
            logger.error(f"Error in summarize_ticket: {str(e)}")
            return {"summary": f"Customer submitted a {state.get('category', 'support')} ticket requiring attention."}
    
    def route_ticket(self, state: TicketState) -> Dict[str, Any]:
        """
        Node 4: Route the ticket to the appropriate email based on category and priority.
        
        Args:
            state: Current ticket state
            
        Returns:
            Updated state with routing_email field
        """
        try:
            logger.info("Starting ticket routing")
            
            category = state.get("category", "General Inquiry")
            priority = state.get("priority", "Medium")
            
            # Get routing email based on rules
            routing_key = (category, priority)
            routing_email = self.routing_rules.get(routing_key, "support@company.com")
            
            logger.info(f"Ticket routed to: {routing_email} (Category: {category}, Priority: {priority})")
            return {"routing_email": routing_email}
            
        except Exception as e:
            logger.error(f"Error in route_ticket: {str(e)}")
            return {"routing_email": "support@company.com"}  # Default fallback
    
    def generate_acknowledgment(self, state: TicketState) -> Dict[str, Any]:
        """
        Node 5: Generate an acknowledgment email snippet.
        
        Args:
            state: Current ticket state
            
        Returns:
            Updated state with acknowledgment field
        """
        try:
            logger.info("Starting acknowledgment generation")
            
            system_prompt = """You are a customer service email writer. Create a professional 1-2 sentence acknowledgment snippet for a support ticket.

Guidelines:
- Keep it to 1-2 sentences maximum
- Be professional and empathetic
- Reference the main issue from the summary
- Indicate that the ticket has been received and routed appropriately
- Don't make specific promises about resolution time
- Use a warm, helpful tone

Examples:
- "Thank you for contacting us about your login issues. Your ticket has been forwarded to our technical team for immediate assistance."
- "We've received your billing inquiry regarding the duplicate charge. Our billing specialists will review your account and respond shortly."
- "Your request about CRM integration options has been received. Our support team will provide you with detailed information about available solutions."

Respond with ONLY the 1-2 sentence acknowledgment."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Create an acknowledgment for this ticket:\nSummary: {state['summary']}\nCategory: {state['category']}\nPriority: {state['priority']}")
            ]
            
            response = self.llm.invoke(messages)
            acknowledgment = str(response.content).strip()
            
            logger.info("Acknowledgment generated successfully")
            return {"acknowledgment": acknowledgment}
            
        except Exception as e:
            logger.error(f"Error in generate_acknowledgment: {str(e)}")
            return {"acknowledgment": "Thank you for contacting us. Your ticket has been received and will be addressed by our support team."}
    
    def _build_graph(self):
        """
        Build the LangGraph workflow with sequential node connections.
        
        Returns:
            Compiled StateGraph
        """
        try:
            logger.info("Building LangGraph workflow")
            
            # Create the graph
            workflow = StateGraph(TicketState)
            
            # Add nodes
            workflow.add_node("classify", self.classify_ticket)
            workflow.add_node("prioritize", self.detect_priority)
            workflow.add_node("summarize", self.summarize_ticket)
            workflow.add_node("route", self.route_ticket)
            workflow.add_node("acknowledge", self.generate_acknowledgment)
            
            # Define the sequential flow
            workflow.add_edge(START, "classify")
            workflow.add_edge("classify", "prioritize")
            workflow.add_edge("prioritize", "summarize")
            workflow.add_edge("summarize", "route")
            workflow.add_edge("route", "acknowledge")
            workflow.add_edge("acknowledge", END)
            
            # Compile with checkpointer
            memory = InMemorySaver()
            compiled_graph = workflow.compile(checkpointer=memory)
            
            logger.info("LangGraph workflow built successfully")
            return compiled_graph
            
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise
    
    def process_ticket(self, ticket_text: str, config: Optional[Dict[str, Any]] = None) -> TicketState:
        """
        Process a support ticket through the complete triage workflow.
        
        Args:
            ticket_text: Raw customer support ticket text
            config: Optional configuration for the graph execution
            
        Returns:
            Complete TicketState with all fields populated
        """
        try:
            logger.info("Processing support ticket")
            
            # Initialize state
            initial_state = TicketState(
                ticket_text=ticket_text,
                category="",
                priority="",
                summary="",
                routing_email="",
                acknowledgment=""
            )
            
            # Use default config if none provided
            if config is None:
                config = {"configurable": {"thread_id": "default"}}
            
            # Execute the workflow
            result = self.graph.invoke(initial_state, config=config)
            
            logger.info("Ticket processing completed successfully")
            logger.info(f"Final routing: {result['routing_email']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing ticket: {str(e)}")
            raise
    
    def get_routing_rules(self) -> Dict[tuple, str]:
        """
        Get the current email routing rules.
        
        Returns:
            Dictionary of routing rules
        """
        return self.routing_rules.copy()


def main():
    """
    Example usage of the Support Triage Agent.
    """
    try:
        # Initialize the agent
        agent = SupportTriageAgent()
        
        # Example ticket
        sample_ticket = """
        Hi there,
        
        I'm having a critical issue with my account. I can't log in at all and this is blocking 
        my entire team from working. We have a major presentation tomorrow and need access 
        immediately. This started happening after the maintenance window last night.
        
        Please help ASAP!
        
        Thanks,
        John Smith
        """
        
        # Process the ticket
        print("Processing support ticket...")
        result = agent.process_ticket(sample_ticket)
        
        # Display results
        print("\n" + "="*50)
        print("SUPPORT TICKET TRIAGE RESULTS")
        print("="*50)
        print(f"Category: {result['category']}")
        print(f"Priority: {result['priority']}")
        print(f"Summary: {result['summary']}")
        print(f"Routing Email: {result['routing_email']}")
        print(f"Acknowledgment: {result['acknowledgment']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()











