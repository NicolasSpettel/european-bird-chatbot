import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agents.bird_qa_agent import BirdQAAgent
import logging

logging.basicConfig(level=logging.INFO)

def test_bird_agent():
    """Test the Bird Q&A Agent"""
    print("Initializing Bird Q&A Agent...")
    
    try:
        agent = BirdQAAgent()
        print("Agent initialized successfully")
        
        # Test questions
        test_questions = [
            "can we talk about crows",
            "what is their primary diet?",
            "what does the barn owl look like?",
            "is the barn owl the same as the tawny owl?"
        ]
        
        print("\nTesting agent with questions:")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 30)
            
            response = agent.ask(question)
            
            if response["error"]:
                print(f"ERROR: {response['answer']}")
            else:
                print(f"Answer: {response['answer']}")
                
        print("\n" + "=" * 50)
        print("Agent testing complete!")
        
    except Exception as e:
        print(f"Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bird_agent()
