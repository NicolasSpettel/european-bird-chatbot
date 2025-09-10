import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.tools.audio_processor import AudioProcessor
from src.agents.bird_qa_agent import BirdQAAgent
import logging

logging.basicConfig(level=logging.INFO)

def run_audio_test():
    """
    Tests the audio processing and agent pipeline with a real audio file.
    """
    audio_file = "./tests/test_audio/barn_or_tawny_owl.wav"

    if not os.path.exists(audio_file):
        print(f"❌ Error: The audio file '{audio_file}' was not found.")
        print("Please ensure your folder structure is correct and the file exists.")
        return

    print(f"🎵 Found audio file: {audio_file}")
    
    try:
        # Step 1: Transcribe the audio
        print("\n🔄 Transcribing audio...")
        processor = AudioProcessor()
        transcription = processor.transcribe_audio(audio_file)
        
        if not transcription:
            print("❌ Transcription failed or returned an empty result.")
            return

        print(f"✅ Transcription successful: '{transcription}'")
        
        # Step 2: Process the transcription with the Bird Q&A Agent
        print("\n🤖 Sending query to Bird Q&A Agent...")
        agent = BirdQAAgent()
        response = agent.process_audio_query(audio_file)
        
        print("\n📋 Agent Response:")
        if response.get('error'):
            print(f"❌ An error occurred: {response['answer']}")
        else:
            print(f"✅ Agent returned a successful response.")
            print(f"Answer: {response['answer']}")
            if 'images' in response and response['images']:
                print(f"Image URL: {response['images'][0]}")

    except Exception as e:
        print(f"❌ An unexpected error occurred during the test: {e}")
        logging.error("Test failed due to an exception.", exc_info=True)

if __name__ == "__main__":
    print("🐦 Running simplified audio test...")
    print("-" * 35)
    run_audio_test()
    print("-" * 35)
    print("Test finished.")