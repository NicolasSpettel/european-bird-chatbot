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

        # 🔥 Reset memory before testing
        agent.clear_memory()
        print("Agent memory reset.\n")
        
        # Test questions
        test_questions = [
            "What's the most common bird in Europe?",
            "Tell me about migratory birds in Europe.",
            "What bird has a red chest and is often seen in gardens?",
            "Which European birds are known for their unique songs or calls?",
            "What's the difference between a crow and a raven?",
            "What does the Eurasian Jay look like?",
            "What does the Eurasian Jay eat?",
            "How does the Eurasian Jay's behavior change throughout the seasons?",
            "Where can I typically find the Eurasian Jay in Europe?",
            "Can the Eurasian Jay mimic other birds or sounds?",
            "What's the best time of day to go birdwatching?",
            "What kind of binoculars should I buy for a beginner?",
            "Can you recommend a good European dish?",
            "What are the best hiking trails in the Dolomites?",
            "How does a jet engine work?"
        ]
        
        print("\nTesting agent with questions:")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 30)
            
            response = agent.ask(question)
            
            if response["error"]:
                print(f"ERROR: {response['response']}")
            else:
                print(f"Answer: {response['response']}")
                
        print("\n" + "=" * 50)
        print("Agent testing complete!")
        
    except Exception as e:
        print(f"Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bird_agent()


