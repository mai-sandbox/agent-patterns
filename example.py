#!/usr/bin/env python3
"""
Example CLI interface for the form-filling agent.

This script demonstrates how to use the form-filling agent to process forms
section by section with human-in-the-loop validation and persistent memory.
"""

import asyncio
import os
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from agent import app, DEFAULT_FORM_SECTIONS, FORM_FIELDS
from state import FormFillingState, create_initial_state, get_completion_percentage


def print_banner():
    """Print a welcome banner for the form-filling agent."""
    print("=" * 60)
    print("🔥 FORM-FILLING AGENT DEMO 🔥")
    print("=" * 60)
    print("This demo shows how to use the LangGraph form-filling agent")
    print("to process forms section by section with validation.")
    print("=" * 60)
    print()


def print_section_info(section_name: str):
    """Print information about the current section."""
    section_fields = FORM_FIELDS.get(section_name, {})
    print(f"\n📋 **{section_name.replace('_', ' ').title()} Section**")
    print("-" * 40)
    print("Fields to complete:")
    
    for field_name, field_config in section_fields.items():
        required = " (required)" if field_config.get("required", False) else " (optional)"
        print(f"  • {field_name.replace('_', ' ').title()}: {field_config['description']}{required}")
    print()


def print_progress(state: FormFillingState):
    """Print current form progress."""
    completed = len(state.get("sections_completed", []))
    total = state.get("total_sections", 0)
    percentage = get_completion_percentage(state)
    current = state.get("current_section", "")
    
    print(f"\n📊 **Progress:** {completed}/{total} sections completed ({percentage:.1f}%)")
    if current:
        print(f"🔄 **Current Section:** {current.replace('_', ' ').title()}")
    print()


def handle_human_input(interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle human-in-the-loop input during form validation.
    
    Args:
        interrupt_data: Data from the interrupt() call
        
    Returns:
        Dict[str, Any]: Human response
    """
    print("\n" + "="*50)
    print("🤖 **HUMAN VALIDATION REQUIRED** 🤖")
    print("="*50)
    
    section = interrupt_data.get("section", "")
    data = interrupt_data.get("data", {})
    validation_errors = interrupt_data.get("validation_errors", [])
    question = interrupt_data.get("question", "Please review the data.")
    
    print(f"\n**Section:** {section.replace('_', ' ').title()}")
    print(f"**Question:** {question}")
    
    # Display the data for review
    print("\n**Data to Review:**")
    for field_name, value in data.items():
        if not field_name.startswith("_"):
            print(f"  • {field_name.replace('_', ' ').title()}: {value}")
    
    # Display validation errors if any
    if validation_errors:
        print(f"\n⚠️ **Validation Issues ({len(validation_errors)}):**")
        for error in validation_errors:
            print(f"  • {error}")
    
    # Get human response
    print("\n**Options:**")
    print("  1. Type 'yes' or 'y' to approve the data")
    print("  2. Type 'no' or 'n' to make corrections")
    print("  3. Type 'skip' to skip this section")
    print("  4. Type 'restart' to restart this section")
    
    while True:
        response = input("\n👤 Your choice: ").strip().lower()
        
        if response in ["yes", "y", "approve", "correct"]:
            return {"approved": "yes"}
        
        elif response in ["no", "n"]:
            print("\n**Making Corrections:**")
            corrections = {}
            
            for field_name in data.keys():
                if not field_name.startswith("_"):
                    current_value = data[field_name]
                    print(f"\nCurrent {field_name.replace('_', ' ')}: {current_value}")
                    new_value = input(f"New value (press Enter to keep current): ").strip()
                    
                    if new_value:
                        # Try to convert to appropriate type
                        if field_name == "years_experience":
                            try:
                                corrections[field_name] = int(new_value)
                            except ValueError:
                                corrections[field_name] = new_value
                        elif field_name == "newsletter":
                            corrections[field_name] = new_value.lower() in ["yes", "y", "true", "1"]
                        else:
                            corrections[field_name] = new_value
            
            feedback = input("\nAny additional feedback (optional): ").strip()
            feedback_list = [feedback] if feedback else []
            
            return {
                "approved": "no",
                "action": "retry",
                "corrections": corrections,
                "feedback": feedback_list
            }
        
        elif response == "skip":
            return {"approved": "no", "action": "skip_section"}
        
        elif response == "restart":
            return {"approved": "no", "action": "restart"}
        
        else:
            print("❌ Invalid choice. Please try again.")


async def run_form_demo():
    """Run the interactive form-filling demo."""
    print_banner()
    
    # Check if API key is configured
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ **Error:** ANTHROPIC_API_KEY not found in environment variables.")
        print("Please set up your .env file with your Anthropic API key.")
        print("See .env.example for the required format.")
        return
    
    print("🚀 Starting form-filling agent...")
    print(f"📝 Form sections: {', '.join([s.replace('_', ' ').title() for s in DEFAULT_FORM_SECTIONS])}")
    
    # Create configuration for this session
    config = {"configurable": {"thread_id": "demo_session_1"}}
    
    # Initialize the form
    initial_input = {
        "messages": [{"role": "user", "content": "I'd like to fill out a form."}]
    }
    
    print("\n🎯 **Starting form-filling process...**")
    
    try:
        # Stream the agent execution
        async for event in app.astream(initial_input, config):
            for node_name, node_output in event.items():
                if "messages" in node_output:
                    # Print the latest message
                    latest_message = node_output["messages"][-1]
                    if hasattr(latest_message, 'content'):
                        print(f"\n🤖 **Agent:** {latest_message.content}")
                    
                    # Print progress if available
                    if "current_section" in node_output:
                        print_progress(node_output)
                
                # Handle interrupts (human-in-the-loop)
                if hasattr(app, 'get_state'):
                    state = app.get_state(config)
                    if state.next and 'interrupt' in str(state.next):
                        # This is a simplified interrupt handler
                        # In a real implementation, you'd handle the interrupt properly
                        print("\n⏸️ **Interrupt detected - human input required**")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
        return
    except Exception as e:
        print(f"\n❌ **Error during demo:** {str(e)}")
        print("Please check your configuration and try again.")
        return
    
    print("\n✅ **Demo completed!**")
    print("Thank you for trying the form-filling agent!")


def run_simple_example():
    """Run a simple synchronous example."""
    print_banner()
    
    # Check if API key is configured
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ **Error:** ANTHROPIC_API_KEY not found in environment variables.")
        print("Please set up your .env file with your Anthropic API key.")
        print("See .env.example for the required format.")
        return
    
    print("🚀 Running simple form-filling example...")
    
    # Create configuration for this session
    config = {"configurable": {"thread_id": "simple_example_1"}}
    
    # Initialize the form
    initial_input = {
        "messages": [{"role": "user", "content": "I'd like to fill out a form."}]
    }
    
    print("\n🎯 **Starting form-filling process...**")
    
    try:
        # Run the agent synchronously
        result = app.invoke(initial_input, config)
        
        # Print the result
        if "messages" in result:
            latest_message = result["messages"][-1]
            if hasattr(latest_message, 'content'):
                print(f"\n🤖 **Agent Response:**")
                print(latest_message.content)
        
        # Print current state
        state = app.get_state(config)
        if state and state.values:
            print_progress(state.values)
            
            current_section = state.values.get("current_section", "")
            if current_section:
                print_section_info(current_section)
    
    except Exception as e:
        print(f"\n❌ **Error during example:** {str(e)}")
        print("Please check your configuration and try again.")
        return
    
    print("\n✅ **Simple example completed!**")
    print("The agent is ready to process your form step by step.")
    print("In a real application, you would continue the conversation by providing form data.")


def main():
    """Main entry point for the example script."""
    # Load environment variables
    load_dotenv()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        print("Running async demo...")
        asyncio.run(run_form_demo())
    else:
        print("Running simple synchronous example...")
        print("(Use --async flag for the full interactive demo)")
        run_simple_example()


if __name__ == "__main__":
    main()
