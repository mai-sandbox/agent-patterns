#!/usr/bin/env python3
"""Test script to check ChatAnthropic API parameters."""

from langchain_anthropic import ChatAnthropic
import inspect

# Get the signature of ChatAnthropic.__init__
sig = inspect.signature(ChatAnthropic.__init__)
print("ChatAnthropic.__init__ parameters:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  {param_name}: {param.annotation} = {param.default}")
