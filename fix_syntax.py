#!/usr/bin/env python3

# Fix the unterminated f-string in app.py
with open('app.py', 'r') as f:
    content = f.read()

# Replace the broken f-string
content = content.replace('st.info(f"**Current System Prompt:**', 'st.info(f"**Current System Prompt:**\\n\\n{st.session_state.system_prompt}")')

with open('app.py', 'w') as f:
    f.write(content)


