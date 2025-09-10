from flask import Flask, render_template, request, jsonify
from src.agents.bird_qa_agent import BirdQAAgent
import tempfile
import os
import logging

app = Flask(__name__)
bird_agent = BirdQAAgent()
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    print("=== NEW REQUEST ===")
    print("Received request:", request.json)
    user_input = request.json.get('message')
    print("User input:", user_input)
    
    try:
        # Get the response from bird agent
        response = bird_agent.ask(user_input)
        print("Bird agent response:", response)
        
        # Extract the actual answer and image
        actual_response = response.get('answer', 'No response available')
        actual_images = response.get('images', [])
        actual_image = actual_images[0] if actual_images else None
        
        print(f"Final response: {actual_response}")
        print(f"Final image: {actual_image}")
        print("=== END REQUEST ===")
        
        return jsonify({
            'response': actual_response,
            'image_url': actual_image
        })
        
    except Exception as e:
        print(f"Error in Flask route: {e}")
        return jsonify({
            'response': 'Sorry, I encountered an error processing your request.',
            'image_url': None
        }), 500

@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name
        
        response = bird_agent.process_audio_query(temp_path)
        os.unlink(temp_path)
        
        return jsonify({
            'response': response.get('answer', 'No response'),
            'image_url': response['images'][0] if response['images'] else None,
            'transcription': response.get('transcription', None)
        })
    except Exception as e:
        logger.error(f"Error processing audio upload: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)