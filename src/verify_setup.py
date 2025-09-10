# src/verify_setup.py
"""
Verify your setup is working - NO TEST DATA, just check connections
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def verify_imports():
    """Verify all imports work"""
    print("🔧 Verifying imports...")
    
    try:
        from src.config import Config
        print("✅ Config imported")
        
        from src.database.chroma_client import ChromaClient
        print("✅ ChromaClient imported")
        
        import openai
        print("✅ OpenAI imported")
        
        import chromadb
        print("✅ ChromaDB imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def verify_config():
    """Verify configuration"""
    print("\n🔧 Verifying configuration...")
    
    try:
        from src.config import Config
        
        if Config.OPENAI_API_KEY:
            if Config.OPENAI_API_KEY.startswith('sk-'):
                print("✅ OpenAI API key found and looks valid")
            else:
                print("⚠️ OpenAI API key found but doesn't look like OpenAI format")
        else:
            print("❌ No OpenAI API key found in .env file")
            return False
            
        print(f"✅ Database path: {Config.CHROMA_DB_PATH}")
        print(f"✅ Debug mode: {Config.DEBUG}")
        
        return True
    except Exception as e:
        print(f"❌ Config verification failed: {e}")
        return False

def verify_chromadb():
    """Verify ChromaDB initialization"""
    print("\n🔧 Verifying ChromaDB...")
    
    try:
        from src.database.chroma_client import ChromaClient
        
        client = ChromaClient()
        print("✅ ChromaDB client created successfully")
        print(f"✅ Collections available: {list(client.collections.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB verification failed: {e}")
        return False

def verify_openai():
    """Verify OpenAI connection"""
    print("\n🔧 Verifying OpenAI connection...")
    
    try:
        from openai import OpenAI
        from src.config import Config
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Simple test
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say: Connection working"}],
            max_tokens=5
        )
        
        print(f"✅ OpenAI response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI verification failed: {e}")
        return False

def main():
    """Run all verifications"""
    print("🚀 European Bird Chatbot - Setup Verification\n")
    
    results = []
    
    # Check each component
    results.append(verify_imports())
    results.append(verify_config())
    results.append(verify_chromadb())
    results.append(verify_openai())
    
    print("\n" + "="*50)
    
    if all(results):
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Ready to proceed with data collection")
    else:
        print("❌ Some verifications failed")
        print("🔧 Fix the issues above before continuing")
        
        # Give specific help
        if not results[0]:
            print("\n💡 Import issues: Check your file structure and __init__.py files")
        if not results[1]:
            print("\n💡 Config issues: Check your .env file has OPENAI_API_KEY=sk-...")
        if not results[2]:
            print("\n💡 ChromaDB issues: Make sure chromadb is installed: pip install chromadb")
        if not results[3]:
            print("\n💡 OpenAI issues: Check your API key is valid and has credits")

if __name__ == "__main__":
    main()