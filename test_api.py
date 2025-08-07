#!/usr/bin/env python3
"""Test script to check ChatAnthropic API parameters."""

from langchain_anthropic import ChatAnthropic
import os

# Try to create an instance to see what parameters work
try:
    # Test with minimal parameters
    llm = ChatAnthropic()
    print("ChatAnthropic created successfully with no parameters")
except Exception as e:
    print(f"Error creating ChatAnthropic with no parameters: {e}")

# Check the class attributes and methods
print("\nChatAnthropic attributes:")
for attr in dir(ChatAnthropic):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Try to see the docstring
print(f"\nChatAnthropic docstring:\n{ChatAnthropic.__doc__}")

# Check if there's a model_name attribute
try:
    llm = ChatAnthropic(model="claude-3-haiku-20240307")
    print("Successfully created with 'model' parameter")
except Exception as e:
    print(f"Error with 'model' parameter: {e}")

try:
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    print("Successfully created with 'model_name' parameter")
except Exception as e:
    print(f"Error with 'model_name' parameter: {e}")

