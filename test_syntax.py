#!/usr/bin/env python3
"""
Simple syntax test for the updated macos_tts_hotkey.py
"""

try:
    import ast
    
    # Read and parse the file
    with open('macos_tts_hotkey.py', 'r') as f:
        source_code = f.read()
    
    # Try to parse it
    ast.parse(source_code)
    print("✅ Syntax check passed!")
    
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")
except Exception as e:
    print(f"❌ Error: {e}")