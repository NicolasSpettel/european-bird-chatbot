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
    print("Received request:", request.json)
    user_input = request.json.get('message')
    print("User input:", user_input)
    
    # Debug: Get raw agent response first
    raw_agent_response = bird_agent.agent.invoke({"input": user_input})
    print("Raw agent response:", raw_agent_response)
    
    response = bird_agent.ask(user_input)
    print("Processed agent response:", response)
    
    # Parse the nested response format
    actual_response = "No response"
    actual_image = None
    
    if isinstance(response, dict) and 'answer' in response:
        answer_str = response['answer']
        print(f"Answer string: {answer_str}")
        
        # Look for images in the original response dict
        if 'images' in response and response['images']:
            actual_image = response['images'][0]
            print(f"Found image in response['images']: {actual_image}")
        
        # Try to parse if it's a string representation of a dict
        try:
            import ast
            import re
            
            if answer_str.startswith("{'") and answer_str.endswith("'}"):
                parsed = ast.literal_eval(answer_str)
                actual_response = parsed.get('response', answer_str)
                if not actual_image:
                    actual_image = parsed.get('image_url', None)
            else:
                actual_response = answer_str
                # Extract URL from plain text if present
                if not actual_image:
                    url_match = re.search(r'https://[^\s\'")}]+\.(jpg|jpeg|png|gif)', answer_str)
                    if url_match:
                        actual_image = url_match.group(0)
                        print(f"Extracted image from text: {actual_image}")
        except Exception as e:
            print(f"Parse error: {e}")
            actual_response = answer_str
    
    print(f"Final response: {actual_response}")
    print(f"Final image: {actual_image}")
    
    return jsonify({
        'response': actual_response,
        'image_url': actual_image
    })

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