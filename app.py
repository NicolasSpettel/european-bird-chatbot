import json
import logging
import re
from flask import Flask, render_template, request, jsonify, Response
import requests
from flask_cors import CORS
from dotenv import load_dotenv
from src.agents.bird_qa_agent import BirdQAAgent
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

logger = logging.getLogger(__name__)

try:
    bird_agent = BirdQAAgent()
    logger.info("Bird QA Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Bird QA Agent: {e}")
    bird_agent = None


def strip_markdown_links(text: str) -> str:
    """Remove Markdown image and link syntax from text."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    return ' '.join(text.split())


@app.route('/reset_memory', methods=['POST'])
def reset_memory():
    """Reset the bird agent's conversation memory."""
    if bird_agent is None:
        return jsonify({'error': 'System not properly initialized'}), 500

    try:
        bird_agent.clear_memory()
        logger.info("Memory reset via frontend button")
        return jsonify({'status': 'success', 'message': 'Memory reset. Fresh start!'})
    except Exception as e:
        logger.error(f"Error resetting memory: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/stream_audio', methods=['GET'])
def stream_audio():
    """Stream audio files from allowed domains (security proxy)."""
    audio_url = request.args.get('url')
    if not audio_url:
        return "URL parameter missing", 400
    
    allowed_domains = ('https://xeno-canto.org/',)
    if not audio_url.startswith(allowed_domains):
        logger.warning(f"Unauthorized URL access attempt: {audio_url}")
        return "Unauthorized URL", 403

    try:
        req = requests.get(audio_url, stream=True, timeout=10)
        req.raise_for_status()
        
        return Response(
            req.iter_content(chunk_size=1024), 
            mimetype=req.headers.get('Content-Type')
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error streaming audio from {audio_url}: {e}")
        return "Error streaming audio", 500


@app.route('/')
def home():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return "OK", 200


@app.route('/ask', methods=['POST'])
def ask():
    """Process text-based bird queries."""
    if bird_agent is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_input = data['message']
        logger.info(f"Received text query: {user_input}")

        response_data = bird_agent.ask(user_input)
        logger.info(f"Agent response: {response_data}")

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            'response': 'An internal server error occurred.',
            'birds': [],
            'error': True
        }), 500


@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    """Process audio-based bird queries with transcription."""
    if bird_agent is None:
        return jsonify({'error': 'System not properly initialized'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        transcription_response = bird_agent.process_audio_bytes(
            audio_bytes, 
            filename=audio_file.filename
        )
        logger.info(f"Audio transcription response: {transcription_response}")

        if transcription_response.get('error'):
            return jsonify({
                'error': transcription_response.get('message', 'Failed to process audio'),
                'transcription': transcription_response.get('transcription', '')
            }), 500

        transcription = transcription_response.get('transcription', '')
        logger.info(f"Transcription: {transcription}")

        agent_response_data = bird_agent.ask(transcription)
        logger.info(f"Agent response to transcription: {agent_response_data}")

        return jsonify({
            'response': agent_response_data.get('response', 'No response.'),
            'birds': agent_response_data.get('birds', []),
            'transcription': transcription,
            'error': agent_response_data.get('error', False)
        })

    except Exception as e:
        logger.error(f"Error processing audio upload: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with JSON response."""
    return jsonify({'error': 'Endpoint not found'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)