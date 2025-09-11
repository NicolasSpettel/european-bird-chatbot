import logging
import sys
import os
from pathlib import Path

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.bird_qa_agent import BirdQAAgent

# Configure logging to see what the agent is doing
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_audio_test(audio_file_path: str):
    """
    Tests the BirdQAAgent's audio processing functionality.
    
    Args:
        audio_file_path: The path to the audio file to be processed.
    """
    try:
        logger.info(f"Loading audio from: {audio_file_path}")
        
        # Read the audio file into a byte stream
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()

        logger.info("Initializing the Bird Q&A Agent...")
        agent = BirdQAAgent()
        
        # Use the correct method name `process_audio_bytes`
        logger.info("Sending audio bytes to agent for processing...")
        response = agent.process_audio_bytes(audio_bytes, filename=Path(audio_file_path).name)
        
        # Display the results
        if not response['error']:
            print("✅ Test successful:")
            print(f"   Transcription: '{response.get('transcription', 'N/A')}'")
            print(f"   Agent's Answer: {response['answer']}")
        else:
            print(f"❌ Test failed due to an error: {response['answer']}")
        
    except FileNotFoundError:
        logger.error(f"Audio file not found at: {audio_file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)

if __name__ == "__main__":
    test_audio_file = "./tests/test_audio/barn_or_tawny_owl.wav"
    run_audio_test(test_audio_file)
