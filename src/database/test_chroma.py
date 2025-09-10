# src/database/test_chroma.py
"""
Test script for ChromaDB setup - run this to verify everything works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.database.chroma_client import ChromaClient
from src.config import Config
import openai

def test_chroma_setup():
    """Test ChromaDB basic operations"""
    print("🔧 Testing ChromaDB setup...")
    
    try:
        # Initialize client
        chroma_client = ChromaClient()
        print("✅ ChromaDB client created successfully")
        
        # Test adding a document
        test_documents = [
            "The European Robin is a small bird with a distinctive red breast. It's commonly found in gardens and woodlands across Europe.",
            "The House Sparrow is a small brown and gray bird that lives close to human settlements."
        ]
        
        test_metadata = [
            {"species": "European Robin", "family": "Turdidae", "region": "Europe"},
            {"species": "House Sparrow", "family": "Passeridae", "region": "Europe"}
        ]
        
        test_ids = ["robin_001", "sparrow_001"]
        
        # Add to collection
        chroma_client.add_bird_data(
            collection_name="bird_descriptions",
            documents=test_documents,
            metadata=test_metadata,
            ids=test_ids
        )
        print("✅ Successfully added test documents to ChromaDB")
        
        # Test search
        results = chroma_client.search_birds(
            collection_name="bird_descriptions",
            query="red breast bird",
            n_results=2
        )
        
        if results and len(results['documents'][0]) > 0:
            print("✅ Search working! Found:", results['metadatas'][0][0]['species'])
        else:
            print("❌ Search returned no results")
            
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🔧 Testing OpenAI connection...")
    
    try:
        openai.api_key = Config.OPENAI_API_KEY
        
        # Test basic completion
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'OpenAI connection works!'"}],
            max_tokens=10
        )
        
        print(f"✅ OpenAI response: {response.choices[0].message.content}")
        
        # Test embeddings
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="Test embedding"
        )
        
        if len(embedding_response.data[0].embedding) > 0:
            print("✅ Embeddings working!")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        print("📝 Make sure your OPENAI_API_KEY is set correctly in .env file")
        return False

if __name__ == "__main__":
    print("🚀 Running database and API tests...\n")
    
    # Test ChromaDB
    chroma_success = test_chroma_setup()
    
    # Test OpenAI
    openai_success = test_openai_connection()
    
    if chroma_success and openai_success:
        print("\n🎉 ALL TESTS PASSED! Ready for next hour.")
    else:
        print("\n❌ Some tests failed. Fix these before proceeding.")

