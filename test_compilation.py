#!/usr/bin/env python3
"""
Test script to verify that the LangGraph form filling agent compiles successfully.
"""

import os
import sys

def test_agent_compilation():
    """Test that the agent compiles without runtime errors."""
    try:
        # Set a dummy API key to avoid initialization errors
        os.environ['ANTHROPIC_API_KEY'] = 'dummy_key_for_testing'
        
        print("Testing agent compilation...")
        
        # Import the compiled graph
        from agent import app
        
        print("✅ Agent imported successfully!")
        print(f"✅ Graph type: {type(app)}")
        print(f"✅ Graph class: {app.__class__.__name__}")
        
        # Test basic graph properties
        if hasattr(app, 'nodes'):
            print(f"✅ Graph has nodes: {list(app.nodes.keys()) if app.nodes else 'None'}")
        
        if hasattr(app, 'edges'):
            print(f"✅ Graph has edges configured")
        
        print("✅ Agent compilation test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

if __name__ == "__main__":
    success = test_agent_compilation()
    sys.exit(0 if success else 1)
