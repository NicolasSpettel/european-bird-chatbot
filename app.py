from flask import Flask, render_template, request, jsonify
from src.agents.bird_qa_agent import BirdQAAgent

# Initialize the Flask app and your agent
app = Flask(__name__)
bird_agent = BirdQAAgent()

# This route serves the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# This route handles the chat requests
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message')
    if user_input:
        response = bird_agent.ask(user_input)
        return jsonify({'response': response['answer']})
    return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    # Run the app on all network interfaces
    app.run(host='0.0.0.0', port=5000)