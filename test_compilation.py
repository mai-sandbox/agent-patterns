#!/usr/bin/env python3
"""
Test script to validate agent compilation and basic functionality.
"""

def test_agent_compilation():
    """Test that the agent compiles successfully."""
    try:
        from agent import app
        print("✅ Agent compilation successful!")
        return True
    except Exception as e:
        print(f"❌ Agent compilation failed: {e}")
        return False

def test_tools_import():
    """Test that tools can be imported successfully."""
    try:
        from tools import tools, web_search, summarize_webpage
        print("✅ Tools import successful!")
        print(f"   - Found {len(tools)} tools: {[tool.name for tool in tools]}")
        return True
    except Exception as e:
        print(f"❌ Tools import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic agent functionality without API calls."""
    try:
        from agent import app
        
        # Test that the graph has the expected structure
        graph = app.get_graph()
        nodes = list(graph.nodes.keys())
        print(f"✅ Graph structure validation successful!")
        print(f"   - Nodes: {nodes}")
        
        # Verify expected nodes exist
        expected_nodes = ["chatbot", "tools"]
        for node in expected_nodes:
            if node in nodes:
                print(f"   - ✅ Node '{node}' found")
            else:
                print(f"   - ❌ Node '{node}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Running agent validation tests...")
    print("=" * 50)
    
    tests = [
        test_tools_import,
        test_agent_compilation,
        test_basic_functionality,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        exit(0)
    else:
        print("⚠️  Some validation tests failed!")
        exit(1)
