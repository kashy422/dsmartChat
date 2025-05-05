"""
Test script to check if agent.py can be imported properly.
"""

try:
    print("Attempting to import from agent.py...")
    from src.agent import chat_engine
    print("✅ Successfully imported chat_engine from agent.py")
    
    # Try to initialize the chat engine
    try:
        engine = chat_engine()
        print("✅ Successfully initialized chat engine")
    except Exception as e:
        print(f"❌ Error initializing chat engine: {str(e)}")
    
except SyntaxError as e:
    print(f"❌ Syntax error: {str(e)}")
    print(f"Line {e.lineno}, column {e.offset}")
    print(f"Text: {e.text}")
    
except Exception as e:
    print(f"❌ Other error: {str(e)}") 