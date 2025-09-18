# European Bird AI Assistant 🦜
An intelligent, conversational AI assistant specializing in European birds. This project leverages a multi-tool agentic architecture to provide users with detailed bird information, visual aids, and auditory samples.

Presentation link: https://docs.google.com/presentation/d/1T6uYEX4yAFgKWyVu-DLxXJyrlY9KuG7dqSyW540P2L0/edit?slide=id.g3807a635d55_0_24#slide=id.g3807a635d55_0_24

# ✨ Features
Intelligent Q&A: Ask about specific birds ("What does a European Robin look like?") or general birdwatching tips ("How do I get started with birding?").

Multimedia Integration: The bot retrieves and displays relevant images and audio clips of birds it discusses.

Conversational Memory: The agent remembers the context of your conversation, allowing for natural, multi-turn interactions.

Voice & Text Input: Interact with the bot via a chat interface or your microphone, with real-time transcription powered by OpenAI's Whisper API.

Specialized Knowledge: The core model has been fine-tuned on a custom dataset of European bird information and birdwatching guides, making it an expert in its domain.

# 🧠 Architecture
The project is built on an agentic framework, where a powerful language model acts as the central "brain," deciding how to respond to user queries.

LLM Agent: We use a fine-tuned version of gpt-4o-mini as our primary agent, configured with a conversational memory (ConversationBufferMemory) to maintain context.

Vector Database: ChromaDB stores vectorized data from multiple sources.

Embedding Model: We fine-tuned the all-MiniLM-L6-v2 model on our custom dataset to improve its semantic understanding of bird-related terms.

# 🛠️ Tools
The LLM uses two custom tools to interact with our specialized knowledge base:

BirdQueryTool: Searches the vector database for comprehensive information about a specific bird species or a descriptive phrase. This tool is essential for providing accurate facts and retrieving images and audio files.

YoutubeQueryTool: Retrieves general birdwatching tips and advice from transcribed YouTube videos.

The agent's system prompt is configured to strictly enforce the use of these tools for any bird-related query, preventing the model from hallucinating information and ensuring a rich, multimedia response.

# 📊 Data Sources
Wikipedia API: Used to collect descriptions and images of ~250 European bird species.

Xeno-Canto: A repository for bird sounds, used to source the audio clips.

YouTube API: Provided transcripts of birdwatching tutorials and guides.

# 🚀 Getting Started
Prerequisites

Before you begin, ensure you have the following software installed on your machine:

Docker: Used to build and run the application in a containerized environment.

AWS CLI: (Optional) If you plan to deploy to AWS, you'll need the AWS Command Line Interface configured.

Python: The project is written in Python. Ensure you have a recent version (3.8+) installed.

You will also need to acquire the following API keys and set them as environment variables:

OPENAI_API_KEY: For gpt-4o-mini and the Whisper API.

HUGGINGFACEHUB_API_TOKEN: For accessing the all-MiniLM-L6-v2 fine-tuned model here: https://huggingface.co/Nicolas-Spettel/bird-qa-model.

We recommend using a .env file to manage these secrets securely.

Clone the repository:

```
git clone https://github.com/NicolasSpettel/european-bird-chatbot.git
cd european-bird-chatbot
```
Run the data collector:

```
python -m src.data.collectors.collect_all_data
```
Populate the database:

```
python -m src.database.populate_db
```
Build the Docker Image:

```
docker build -t bird-chatbot .
```
Run the Container:
```
docker run -p 8000:8000 bird-chatbot
```
Push the image to your container registry (e.g., AWS ECR).

Launch an instance on AWS and deploy the container.

# 📈 Evaluation
Project performance was tracked using LangSmith(https://smith.langchain.com/public/15e7f59c-7fd9-4d8c-a178-98ae49f61718/d), focusing on key metrics like latency and tool-use accuracy. While we observed positive results in real-world testing, the evaluation bot's scores highlighted the difficulty in automatically assessing the quality of conversational AI, where engagement and tone are as important as factual correctness.









