"""State schema definitions for the Support Ticket Triage Agent.

This module defines the TypedDict classes used to manage state throughout
the LangGraph workflow for processing support tickets.
"""

from typing import TypedDict


class TicketState(TypedDict):
    """State schema for the Support Ticket Triage Agent workflow.
    
    This TypedDict defines the complete state structure that flows through
    all nodes in the LangGraph workflow for processing support tickets.
    
    Attributes:
        ticket_text: The original raw customer ticket text input
        category: Ticket classification ('Billing', 'Technical', 'General Inquiry')
        priority: Priority level ('Low', 'Medium', 'High') based on urgency
        summary: One-sentence summary of the ticket content
        email: Routed email address based on category and priority
        acknowledgement: Draft acknowledgement email snippet (1-2 sentences)
    """
    ticket_text: str
    category: str
    priority: str
    summary: str
    email: str
    acknowledgement: str
