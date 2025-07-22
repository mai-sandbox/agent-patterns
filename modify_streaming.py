#!/usr/bin/env python3
import re

# Read the current app.py file
with open('app.py', 'r') as f:
    content = f.read()

# Replace the streaming mode
old_pattern = r'stream_mode="updates"'
new_replacement = 'stream_mode=["values", "updates", "messages"]'
content = re.sub(old_pattern, new_replacement, content)

# Also need to update the loop to handle multiple modes
old_loop = r'for chunk in client\.runs\.stream\('
new_loop = 'for stream_mode, chunk in client.runs.stream('
content = re.sub(old_loop, new_loop, content)

# Add basic handling for different stream modes in the processing logic
old_processing = r'if chunk and hasattr\(chunk, \'data\'\):'
new_processing = '''if stream_mode == 'updates' and chunk and hasattr(chunk, 'data'):'''
content = re.sub(old_processing, new_processing, content)

# Write the modified content back
with open('app.py', 'w') as f:
    f.write(content)

print('Successfully enhanced streaming implementation with multiple modes')

