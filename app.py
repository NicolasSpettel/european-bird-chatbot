import json
import logging
import re
from flask import Flask, render_template, request, jsonify, Response
import requests
from flask_cors import CORS
from dotenv import load_dotenv
from src.agents.bird_qa_agent import BirdQAAgent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Fix CORS - allow all origins for development
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

logger = logging.getLogger(__name__)

# Initialize bird agent with error handling
try:
    bird_agent = BirdQAAgent()
    logger.info("Bird QA Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Bird QA Agent: {e}")
    bird_agent = None



def strip_markdown_links(text: str) -> str:
    """
    Strip Markdown image and link syntax from text.
    Example: "Here is an image: ![alt](url)" becomes "Here is an image: "
    """
    # Remove image syntax: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove link syntax: [text](url)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Remove any extra whitespace
    text = ' '.join(text.split())
    return text

@app.route('/reset_memory', methods=['POST'])
def reset_memory():
    if bird_agent is None:
        return jsonify({'error': 'System not properly initialized'}), 500

    try:
        bird_agent.clear_memory()
        logger.info("Memory reset via frontend button")
        return jsonify({'status': 'success', 'message': 'Memory reset. Fresh start!'})
    except Exception as e:
        logger.error(f"Error resetting memory: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
# The corrected route
@app.route('/stream_audio', methods=['GET'])
def stream_audio():
    audio_url = request.args.get('url')
    if not audio_url:
        return "URL parameter missing", 400

    try:
        # Use requests to get the audio file with streaming enabled
        req = requests.get(audio_url, stream=True)
        req.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Create a Flask Response object
        # The generator expression `req.iter_content(...)` will stream the data
        return Response(req.iter_content(chunk_size=1024), 
                        mimetype=req.headers.get('Content-Type'))

    except requests.exceptions.RequestException as e:
        logger.error(f"Error streaming audio from {audio_url}: {e}")
        return "Error streaming audio", 500
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "agent_initialized": bird_agent is not None
    })

@app.route('/ask', methods=['POST'])
def ask():
    print("=== NEW REQUEST ===")
    
    # Check if agent is initialized
    if bird_agent is None:
        return jsonify({
            'response': 'Sorry, the system is not properly initialized. Please check the server logs.',
            'image_url': '',
            'audio_url': '',
            'error': True
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'response': 'No message provided',
                'image_url': '',
                'audio_url': '',
                'error': True
            }), 400
            
        user_input = data['message']
        logger.info(f"Received text query: {user_input}")

        # The agent is now responsible for returning a consistent dictionary.
        response_data = bird_agent.ask(user_input)
        logger.info(f"Agent response: {response_data}")

        return jsonify({
            'response': response_data.get('answer', response_data.get('description', '')),
            'image_url': response_data.get('image_url', ''),
            'audio_url': response_data.get('audio_url', ''),
            'species': response_data.get('species', ''),
            'error': response_data.get('error', False)
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            'answer': 'An internal server error occurred.',
            'image_url': '',
            'audio_url': '',
            'error': True
        }), 500

@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    if bird_agent is None:
        return jsonify({'error': 'System not properly initialized'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        response = bird_agent.process_audio_bytes(audio_bytes, filename=audio_file.filename)
        logger.info(f"Audio transcription response: {response}")

        if response.get('error'):
            return jsonify({
                'error': response.get('message', 'Failed to process audio'),
                'transcription': response.get('transcription', '')
            }), 500

        # Use the transcription as a text query to the agent
        transcription = response.get('transcription', '')
        logger.info(f"Transcription: {transcription}")

        # Pass the transcription to the agent for a response
        text_response = bird_agent.ask(transcription)
        logger.info(f"Agent response to transcription: {text_response}")

        return jsonify({
            'response': text_response.get('answer', ''),
            'image_url': text_response.get('image_url', ''),
            'audio_url': text_response.get('audio_url', ''),
            'species': text_response.get('species', ''),
            'transcription': transcription,
            'error': False
        })

    except Exception as e:
        logger.error(f"Error processing audio upload: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500


# Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)