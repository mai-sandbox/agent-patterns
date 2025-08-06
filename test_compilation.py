#!/usr/bin/env python3
"""
Test script to verify the LangGraph agent compiles successfully.
"""

import sys

def test_graph_compilation():
    """Test that the agent graph compiles without errors."""
    try:
        import agent
        print("✅ Agent module imported successfully")
        
        # Verify the app is exported
        if hasattr(agent, 'app'):
            print("✅ 'app' variable found in agent module")
            print(f"✅ Graph type: {type(agent.app)}")
            
            # Try to get the graph structure
            try:
                nodes = list(agent.app.get_graph().nodes.keys())
                print(f"✅ Graph nodes: {nodes}")
                print("✅ Graph compiled successfully without errors")
                return True
            except Exception as e:
                print(f"❌ Error accessing graph structure: {e}")
                return False
        else:
            print("❌ 'app' variable not found in agent module")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import agent module: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during compilation test: {e}")
        return False

if __name__ == "__main__":
    success = test_graph_compilation()
    sys.exit(0 if success else 1)
