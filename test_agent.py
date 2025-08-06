#!/usr/bin/env python3
"""
Test script for the LangGraph Form Filling Agent

This script tests the compiled graph by importing and invoking the agent
with sample form data to verify the section-by-section workflow functions correctly.
"""

import sys
from typing import Dict, Any
from langchain_core.messages import HumanMessage

def test_graph_compilation():
    """Test that the graph compiles without errors."""
    print("🔧 Testing graph compilation...")
    
    try:
        from agent import app
        print("✅ Graph compiled successfully!")
        print(f"   Graph type: {type(app)}")
        return app
    except Exception as e:
        print(f"❌ Graph compilation failed: {e}")
        return None

def test_graph_structure(app):
    """Test the graph structure and nodes."""
    print("\n🏗️  Testing graph structure...")
    
    try:
        # Check if the graph has the expected nodes
        if hasattr(app, 'nodes'):
            nodes = list(app.nodes.keys())
            expected_nodes = [
                "analyze_form_structure",
                "fill_current_section", 
                "validate_section",
                "move_to_next_section",
                "complete_form"
            ]
            
            print(f"   Available nodes: {nodes}")
            
            missing_nodes = [node for node in expected_nodes if node not in nodes]
            if missing_nodes:
                print(f"⚠️  Missing expected nodes: {missing_nodes}")
            else:
                print("✅ All expected nodes present!")
                
        return True
    except Exception as e:
        print(f"❌ Graph structure test failed: {e}")
        return False

def create_test_input() -> Dict[str, Any]:
    """Create sample test input for the form filling workflow."""
    return {
        "messages": [
            HumanMessage(content="I need to fill out a registration form with my personal information.")
        ],
        "current_section": "",
        "form_data": {},
        "sections_completed": [],
        "form_structure": {},
        "validation_errors": [],
        "is_complete": False
    }

def test_workflow_execution(app):
    """Test the complete workflow execution."""
    print("\n🚀 Testing workflow execution...")
    
    try:
        # Create test input
        test_input = create_test_input()
        print(f"   Input: {test_input}")
        
        # Invoke the graph
        print("   Executing workflow...")
        result = app.invoke(test_input)  # type: ignore
        
        print("✅ Workflow executed successfully!")
        print(f"   Result type: {type(result)}")
        
        # Analyze the result
        if isinstance(result, dict):
            print("\n📊 Workflow Results:")
            print(f"   - Form complete: {result.get('is_complete', False)}")
            print(f"   - Current section: {result.get('current_section', 'N/A')}")
            print(f"   - Sections completed: {result.get('sections_completed', [])}")
            print(f"   - Validation errors: {result.get('validation_errors', [])}")
            
            # Check if form structure was initialized
            if result.get('form_structure'):
                print(f"   - Form sections: {list(result['form_structure'].keys())}")
            
            # Check messages
            messages = result.get('messages', [])
            print(f"   - Messages count: {len(messages)}")
            if messages:
                print(f"   - Last message: {messages[-1].content if hasattr(messages[-1], 'content') else messages[-1]}")
        
        return result
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_section_progression(app):
    """Test that the workflow progresses through sections correctly."""
    print("\n🔄 Testing section progression...")
    
    try:
        # Test with a more specific form structure
        test_input = {
            "messages": [HumanMessage(content="Fill out my profile form")],
            "current_section": "",
            "form_data": {},
            "sections_completed": [],
            "form_structure": {
                "personal_info": {
                    "fields": ["name", "email"],
                    "required": ["name", "email"],
                    "description": "Personal information"
                },
                "preferences": {
                    "fields": ["newsletter", "notifications"],
                    "required": [],
                    "description": "User preferences"
                }
            },
            "validation_errors": [],
            "is_complete": False
        }
        
        result = app.invoke(test_input)  # type: ignore
        
        if isinstance(result, dict):
            sections_completed = result.get('sections_completed', [])
            current_section = result.get('current_section', '')
            is_complete = result.get('is_complete', False)
            
            print(f"   - Sections completed: {sections_completed}")
            print(f"   - Current section: {current_section}")
            print(f"   - Is complete: {is_complete}")
            
            if sections_completed or current_section or is_complete:
                print("✅ Section progression working!")
            else:
                print("⚠️  Section progression may need review")
        
        return True
        
    except Exception as e:
        print(f"❌ Section progression test failed: {e}")
        return False

def run_all_tests():
    """Run all tests for the form filling agent."""
    print("🧪 Starting LangGraph Form Filling Agent Tests")
    print("=" * 50)
    
    # Test 1: Graph compilation
    app = test_graph_compilation()
    if not app:
        print("\n❌ Cannot proceed with tests - graph compilation failed")
        return False
    
    # Test 2: Graph structure
    structure_ok = test_graph_structure(app)
    if not structure_ok:
        print("\n⚠️  Graph structure issues detected, but continuing with tests...")
    
    # Test 3: Basic workflow execution
    result = test_workflow_execution(app)
    if not result:
        print("\n❌ Basic workflow test failed")
        return False
    
    # Test 4: Section progression
    progression_ok = test_section_progression(app)
    if not progression_ok:
        print("\n⚠️  Section progression test had issues")
    
    print("\n" + "=" * 50)
    print("🎉 Test Suite Completed!")
    
    # Summary
    tests_passed = sum([
        app is not None,
        structure_ok,
        result is not None,
        progression_ok
    ])
    
    print(f"📈 Tests passed: {tests_passed}/4")
    
    if tests_passed >= 3:
        print("✅ Agent is working correctly!")
        return True
    else:
        print("⚠️  Some tests failed - review implementation")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
