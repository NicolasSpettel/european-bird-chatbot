from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.agents.bird_qa_agent import BirdQAAgent
import logging
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Configure logging to see what the agent is doing
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)
bird_agent = BirdQAAgent()
logger = logging.getLogger(__name__)

# This route is no longer used but kept for completeness in the file
@app.route('/')
def home():
    return "Backend is running!"

# Route for handling text-based queries
@app.route('/ask', methods=['POST'])
def ask():
    print("=== NEW REQUEST ===")
    user_input = request.json.get('message')
    logger.info(f"Received text query: {user_input}")
    
    # Get the response from the bird agent
    response = bird_agent.ask(user_input)
    
    logger.info(f"Agent response: {response['answer']}")
    print("=== END REQUEST ===")
    
    # The agent now handles the image markdown, so we just return the full answer.
    return jsonify({
        'response': response['answer'],
        'error': response['error']
    })
    
# Route for handling audio-based queries
@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # The agent's method now handles transcription and error handling
    # The agent's process_audio_bytes method expects the raw bytes
    try:
        audio_bytes = audio_file.read()
        response = bird_agent.process_audio_bytes(audio_bytes, filename=audio_file.filename)
        
        # We only return the final answer and transcription, as the agent
        # now handles all internal logic.
        return jsonify({
            'response': response['answer'],
            'transcription': response['transcription'],
            'error': response['error']
        })
    except Exception as e:
        logger.error(f"Error processing audio upload: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
