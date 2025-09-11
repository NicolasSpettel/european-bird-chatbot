import re
import json
import logging
import os
from typing import List, Dict, Any

from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langsmith import Client
from langchain.memory import ConversationBufferMemory

from src.database.chroma_client import ChromaClient
from src.config import Config
from src.tools.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class BirdQueryTool(BaseTool):
    name: str = "bird_query" 
    description: str = """Search for information about a specific European bird species by name.

    USE THIS TOOL WHEN:
    - User mentions a specific bird name (robin, eagle, sparrow, owl, etc.)
    - User asks "what does [bird species] look like?"
    - User wants identification help for a named species
    - User asks about a specific bird's characteristics, habitat, or behavior

    INPUT: The bird species name (e.g., "european robin", "barn owl", "blue tit")
    OUTPUT: A JSON object containing the bird's species name, a detailed description, and a link to an image."""
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        try:
            logger.info(f"BirdQueryTool searching for: {query}")
            
            results = self.chroma_client.search(
                collection_name="birds",
                query=query,
                n_results=1
            )
            
            if results and results['documents'] and results['documents'][0]:
                doc = results['documents'][0][0]
                metadata = results['metadatas'][0][0]
                
                species = metadata.get('species', 'Unknown')
                image_url = metadata.get('thumbnail', '')
                
                result_data = {
                    "species": species,
                    "description": doc[:1000] if doc else "No description available.",
                    "image_url": image_url if image_url else ""
                }
                
                formatted_output = json.dumps(result_data)
                logger.info(f"BirdQueryTool returning: {formatted_output}")
                return formatted_output
            else:
                result_data = {
                    "species": "Unknown",
                    "description": f"I couldn't find specific information about '{query}'. Try asking about a common European bird.",
                    "image_url": ""
                }
                return json.dumps(result_data)
        
        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            result_data = {
                "species": "Error",
                "description": f"I encountered an error searching for '{query}'. Please try rephrasing your question.",
                "image_url": ""
            }
            return json.dumps(result_data)

class YouTubeQueryTool(BaseTool):
    name: str = "youtube_query"
    description: str = """Search YouTube educational content for birdwatching advice and techniques.

    USE THIS TOOL WHEN:
    - User asks for tips, techniques, or advice
    - User wants equipment recommendations 
    - User asks about birdwatching locations or communities
    - User has "how to" questions about birdwatching
    - User needs beginner guidance

    INPUT: A birdwatching topic or question (e.g., "binocular recommendations", "bird photography tips")
    OUTPUT: Summarized expert advice from educational videos"""
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        try:
            logger.info(f"YouTubeQueryTool searching for: {query}")
            
            results = self.chroma_client.search(
                collection_name="youtube",
                query=query,
                n_results=3
            )
            
            if not results or not results['documents'] or not results['documents'][0]:
                return json.dumps({
                    "summary": f"I couldn't find YouTube content about '{query}'.",
                    "video_count": 0,
                    "video_urls": []
                })
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            combined_content = "\n\n".join([f"From '{meta.get('title', 'Unknown Video')}': {doc[:500]}" 
                                             for doc, meta in zip(documents, metadatas)])
            
            video_urls = [meta.get('url', '') for meta in metadatas if meta.get('url')]
            
            result = {
                "summary": combined_content,
                "video_count": len(documents),
                "video_urls": video_urls[:2]
            }
            
            logger.info(f"YouTubeQueryTool returning {len(documents)} videos")
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"YouTube query failed: {e}")
            return json.dumps({
                "summary": f"I encountered an error searching for '{query}' on YouTube.",
                "video_count": 0,
                "video_urls": []
            })

class BirdQAAgent:
    """Conversational agent that can retrieve bird information from ChromaDB."""
    
    def __init__(self):
        self.chroma_client = ChromaClient()
        self.audio_processor = AudioProcessor()
        
        self.langsmith_client = None
        if Config.LANGSMITH_API_KEY:
            try:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = "european-bird-chatbot"
                os.environ["LANGCHAIN_API_KEY"] = Config.LANGSMITH_API_KEY
                self.langsmith_client = Client(api_key=Config.LANGSMITH_API_KEY)
                logger.info("LangSmith tracking enabled")
            except Exception as e:
                logger.warning(f"LangSmith initialization failed: {e}")
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        self.tools = [
            BirdQueryTool(chroma_client=self.chroma_client),
            YouTubeQueryTool(chroma_client=self.chroma_client)
        ]

    def setup_agent(self):
        system_message = SystemMessage(content="""You are a European birdwatching expert assistant with a conversational memory. Your goal is to provide helpful, concise information to users.

You have access to a specialized tool:
1. bird_query: For specific bird species information. Use this when the user asks about a bird's appearance, habitat, or other facts.

Be conversational and remember previous parts of the conversation. When you use a tool, you must synthesize the tool's output into a readable, natural-sounding response.

If the tool returns an image URL, you must include it in your final answer as a markdown image link, for example: "Here is an image of the European Robin: ![European Robin](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Erithacus_rubecula_with_cocked_head.jpg/330px-Erithacus_rubecula_with_cocked_head.jpg)". Do not just output the URL itself.
""")
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def ask(self, question: str) -> Dict[str, Any]:
        try:
            response = self.agent.invoke({"input": question})
            answer = response['output']

            return {
                "answer": answer,
                "error": False
            }
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": "I encountered an error. Please try rephrasing your question.",
                "error": True
            }

    def process_audio_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        try:
            transcribed_text = self.audio_processor.transcribe_audio_bytes(audio_bytes, filename)
            
            if not transcribed_text:
                return {
                    "answer": "Sorry, I couldn't understand the audio. Please try again with clearer speech.",
                    "error": True
                }
            
            logger.info(f"Transcribed audio bytes: {transcribed_text}")
            
            result = self.ask(transcribed_text)
            
            return {
                "answer": result["answer"],
                "transcription": transcribed_text,
                "error": result["error"]
            }
            
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
            return {
                "answer": f"Error processing audio: {str(e)}",
                "error": True
            }
