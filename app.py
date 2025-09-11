import json
import logging
import re  # Add this import for regex
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from src.agents.bird_qa_agent import BirdQAAgent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:5000"]}})
bird_agent = BirdQAAgent()
logger = logging.getLogger(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return "Backend is running!"

@app.route('/ask', methods=['POST'])
def ask():
    print("=== NEW REQUEST ===")
    user_input = request.json.get('message')
    logger.info(f"Received text query: {user_input}")

    # The agent is now responsible for returning a consistent dictionary.
    response_data = bird_agent.ask(user_input)
    logger.info(f"Agent response: {response_data}")

    # No need for complex if/elif logic here. Just jsonify the result.
    # The agent has already done all the processing.
    return jsonify(response_data)

@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        audio_bytes = audio_file.read()
        response = bird_agent.process_audio_bytes(audio_bytes, filename=audio_file.filename)

        logger.info(f"Audio response type: {type(response)}")
        logger.info(f"Audio response: {response}")

        if isinstance(response, dict):
            if 'answer' in response:
                response['answer'] = strip_markdown_links(response['answer'])
            elif 'description' in response:
                response['description'] = strip_markdown_links(response['description'])

        # Handle the different response types similar to text queries
        if 'answer' in response:
            # Conversational response
            return jsonify({
                'response': response.get('answer', response.get('description', str(response))),
                'image_url': response.get('image_url', ''),
                'audio_url': response.get('audio_url', ''),
                'species': response.get('species', ''),
                'transcription': response.get('transcription', ''),
                'error': response.get('error', False)
            })
        elif 'description' in response:
            # Tool-based response
            return jsonify({
                'response': response['description'],
                'image_url': response.get('image_url', ''),
                'audio_url': response.get('audio_url', ''),
                'species': response.get('species', ''),
                'transcription': response.get('transcription', ''),
                'error': response.get('error', False)
            })
        elif 'summary' in response:
            # YouTube tool response
            youtube_response = response['summary']
            if response.get('video_urls'):
                video_links = "\n\nRelated videos:\n" + "\n".join([f"• {url}" for url in response['video_urls']])
                youtube_response += video_links

            return jsonify({
                'response': youtube_response,
                'image_url': '',
                'audio_url': '',
                'species': '',
                'transcription': response.get('transcription', ''),
                'error': response.get('error', False)
            })
        else:
            # Fallback
            return jsonify({
                'response': str(response),
                'image_url': '',
                'audio_url': '',
                'species': '',
                'transcription': response.get('transcription', ''),
                'error': False
            })
    except Exception as e:
        logger.error(f"Error processing audio upload: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
