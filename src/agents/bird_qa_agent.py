from typing import List, Dict, Any, Optional, Union
import re
import logging
import os

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langsmith import Client

from src.database.chroma_client import ChromaClient
from src.config import Config
from src.tools.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class BirdQueryTool(BaseTool):
    """Tool for searching and retrieving comprehensive information and images for European birds."""
    name: str = "bird_query"
    description: str = "A comprehensive tool for finding detailed descriptions and image URLs for European birds. Input should be a bird name, description, or question about a bird. This tool provides all available information in a single output."
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        try:
            results = self.chroma_client.search_birds(
                collection_name="bird_descriptions",
                query=query,
                n_results=1
            )
            
            if results and results['documents'] and results['documents'][0]:
                doc = results['documents'][0][0]
                metadata = results['metadatas'][0][0]
                
                species = metadata.get('species', 'Unknown')
                image_url = metadata.get('thumbnail', '')
                
                # Return a JSON-like string
                formatted_output = f'{{"species": "{species}", "description": "{doc[:500]}", "image_url": "{image_url}"}}'
                return formatted_output
            else:
                return f"I couldn't find specific information about '{query}'. Try asking about a common European bird."
        
        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            return f"I encountered an error searching for '{query}'. Please try rephrasing your question."

class BirdQAAgent:
    """Main Bird Q&A Agent with audio support and LangSmith tracking"""
    
    def __init__(self):
        self.chroma_client = ChromaClient()
        self.audio_processor = AudioProcessor()
        
        # Initialize LangSmith tracking
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
            model="gpt-3.5-turbo",
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
        self.tools = []
        self.agent = None
        
        self.setup_tools()
        self.setup_agent()
    
    def setup_tools(self):
        """Initialize tools for the agent"""
        self.tools = [
            BirdQueryTool(chroma_client=self.chroma_client),
        ]
        logger.info(f"Initialized {len(self.tools)} tools")
    
    def setup_agent(self):
        """Initialize the LangChain agent"""
        system_message = SystemMessage(content="""You are a European bird expert and guide. Your role is to:

**Work Process:**
1. Use the `bird_query` tool for any user question about a bird.
2. The tool's output will contain all necessary information, including the official species name, a description, and an image URL if available.
3. **Final Response:** Based on the tool's output, create a final, conversational answer. Start with the description, and at the very end, include the image URL on a new line in the format: `Image: [URL]`.

**Tool Rules:**
- `bird_query`: The only tool you have. Use it to retrieve both information and an image URL.
                                       
**Example Conversation Flow:**
Human: Tell me about the European Robin.
You: I'll search for information about the European Robin.
(Agent uses BirdQueryTool)
You: The European Robin is a small passerine bird... (detailed description from search result)... Image: https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Erithacus_rubecula_with_cocked_head.jpg/330px-Erithacus_rubecula_with_cocked_head.jpg

Remember: You specialize in European birds. If asked about non-European species, politely redirect to European alternatives.
""")
        
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=Config.DEBUG,
                max_iterations=3,
                early_stopping_method="generate",
                system_message=system_message
            )
            logger.info("Bird Q&A Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def process_audio_query(self, audio_file_path: str) -> Dict[str, Any]:
        """Process audio input and return response"""
        try:
            # Transcribe audio to text
            transcribed_text = self.audio_processor.transcribe_audio(audio_file_path)
            
            if not transcribed_text:
                return {
                    "answer": "Sorry, I couldn't understand the audio. Please try again with clearer speech.",
                    "images": [],
                    "transcription": None,
                    "error": True
                }
            
            logger.info(f"Transcribed audio: {transcribed_text}")
            
            # Process the transcribed text as a normal query
            result = self.ask(transcribed_text)
            result["transcription"] = transcribed_text
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio query: {e}")
            return {
                "answer": f"Error processing audio: {str(e)}",
                "images": [],
                "transcription": None,
                "error": True
            }
    
    def process_audio_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """Process audio from bytes (for web uploads)"""
        try:
            transcribed_text = self.audio_processor.transcribe_audio_bytes(audio_bytes, filename)
            
            if not transcribed_text:
                return {
                    "answer": "Sorry, I couldn't understand the audio. Please try again with clearer speech.",
                    "images": [],
                    "transcription": None,
                    "error": True
                }
            
            logger.info(f"Transcribed audio bytes: {transcribed_text}")
            
            result = self.ask(transcribed_text)
            result["transcription"] = transcribed_text
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
            return {
                "answer": f"Error processing audio: {str(e)}",
                "images": [],
                "transcription": None,
                "error": True
            }
    
    def ask(self, question: str, input_type: str = "text") -> Dict[str, Any]:
        """Ask the agent a question and return a structured response"""
        try:
            if not self.agent:
                return {
                    "answer": "Agent not properly initialized.",
                    "images": [],
                    "error": True
                }
            
            # Track the query type in LangSmith
            metadata = {"input_type": input_type, "question_length": len(question)}
            
            response = self.agent.invoke(
                {"input": question},
                config={"metadata": metadata} if self.langsmith_client else None
            )['output']
            
            # Extract image URL from response
            match = re.search(r'Image:\s*(https?://\S+)', response)
            
            images = []
            if match:
                images.append(match.group(1))
                answer = response.replace(match.group(0), "").strip()
            else:
                answer = response
            
            return {
                "answer": answer,
                "images": images,
                "error": False
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "images": [],
                "error": True
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        try:
            history = self.memory.chat_memory.messages
            formatted_history = []
            
            for message in history:
                if hasattr(message, 'content'):
                    msg_type = "user" if message.__class__.__name__ == "HumanMessage" else "assistant"
                    formatted_history.append({
                        "type": msg_type,
                        "content": message.content
                    })
            
            return formatted_history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []