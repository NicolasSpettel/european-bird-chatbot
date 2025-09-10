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

class QueryExpander:
    """Utility class to expand queries into multiple variations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def expand_query(self, original_query: str) -> List[str]:
        """Expand a single query into 3 different variations"""
        expansion_prompt = f"""
        Given this birdwatching query: "{original_query}"
        
        Generate 3 different but related search queries that would help find comprehensive information:
        1. A more specific/detailed version
        2. A broader/general version  
        3. A practical/technique-focused version
        
        Return only the 3 queries, one per line, no numbering or extra text.
        Example:
        bird identification techniques
        birdwatching guide for beginners  
        how to identify birds by behavior
        """
        
        try:
            response = self.llm.invoke(expansion_prompt)
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            expanded = queries[:3] if len(queries) >= 3 else [original_query] * 3
            logger.info(f"Expanded '{original_query}' into: {expanded}")
            return expanded
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            # Fallback to manual expansion
            return [
                original_query,
                f"{original_query} tips techniques",
                f"beginner guide {original_query}"
            ]

class BirdQueryTool(BaseTool):
    """Tool for searching and retrieving comprehensive information and images for European birds."""
    name: str = "bird_query"
    description: str = """Use this tool ONLY when the user asks about a SPECIFIC BIRD SPECIES by name (e.g., 'robin', 'eagle', 'sparrow'). 
    This tool provides detailed descriptions and images of European birds. 
    DO NOT use for general birdwatching topics, communities, tips, or techniques."""
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        try:
            results = self.chroma_client.search(
                collection_name="birds",
                query=query,
                n_results=3
            )
            
            if results and results['documents'] and results['documents'][0]:
                doc = results['documents'][0][0]
                metadata = results['metadatas'][0][0]
                
                species = metadata.get('species', 'Unknown')
                image_url = metadata.get('thumbnail', '')
                
                formatted_output = f'{{"species": "{species}", "description": "{doc[:500]}", "image_url": "{image_url}"}}'
                return formatted_output
            else:
                return f"I couldn't find specific information about '{query}'. Try asking about a common European bird."
        
        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            return f"I encountered an error searching for '{query}'. Please try rephrasing your question."

class YouTubeQueryTool(BaseTool):
    """Tool for searching YouTube video transcripts for birdwatching information."""
    name: str = "youtube_query"
    description: str = """Use this tool for general birdwatching topics like: tips, techniques, communities, guides, equipment, 
    identification methods, habitats, behavior, or any non-specific bird questions. 
    Input should be a query related to birdwatching topics (NOT specific bird species)."""
    chroma_client: ChromaClient
    llm: ChatOpenAI
    query_expander: QueryExpander

    def _summarize_content(self, documents: List[str], metadatas: List[dict], original_query: str) -> str:
        """Summarize multiple document excerpts into a coherent response"""
        combined_content = "\n\n".join([f"Video: {meta.get('title', 'Unknown')}\nContent: {doc[:300]}" 
                                       for doc, meta in zip(documents, metadatas)])
        
        summary_prompt = f"""
        Based on these YouTube video excerpts about birdwatching, provide a comprehensive and conversational summary that answers: "{original_query}"
        
        Video Content:
        {combined_content}
        
        Instructions:
        - Synthesize information from all videos, don't just copy text
        - Make it conversational and helpful
        - Focus on practical advice and key insights
        - If videos mention specific techniques or tips, explain them clearly
        - Keep it informative but concise (2-4 sentences)
        - Don't mention "based on videos" - just provide the information naturally
        """
        
        try:
            response = self.llm.invoke(summary_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"I found information from {len(documents)} videos but had trouble summarizing it."

    def _run(self, query: str) -> str:
        try:
            # Expand the query into multiple variations
            expanded_queries = self.query_expander.expand_query(query)
            
            all_documents = []
            all_metadatas = []
            seen_titles = set()
            
            # Search with each expanded query
            for expanded_query in expanded_queries:
                results = self.chroma_client.search(
                    collection_name="youtube",
                    query=expanded_query,
                    n_results=2
                )
                
                if results and results['documents'] and results['documents'][0]:
                    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                        title = metadata.get('title', 'Unknown')
                        if title not in seen_titles:
                            all_documents.append(doc)
                            all_metadatas.append(metadata)
                            seen_titles.add(title)
            
            if not all_documents:
                return f"I couldn't find YouTube content about '{query}'."
            
            # Use top 3 most relevant documents
            documents_to_use = all_documents[:3]
            metadatas_to_use = all_metadatas[:3]
            
            # Generate summary
            summary = self._summarize_content(documents_to_use, metadatas_to_use, query)
            
            # Include video URLs for reference
            video_urls = [meta.get('url', '') for meta in metadatas_to_use if meta.get('url')]
            
            result = {
                "summary": summary,
                "video_count": len(documents_to_use),
                "video_urls": video_urls[:2]
            }
            
            return str(result)
            
        except Exception as e:
            logger.error(f"YouTube query failed: {e}")
            return f"I encountered an error searching for '{query}' on YouTube."

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
        
        self.query_expander = QueryExpander(self.llm)
        
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
            YouTubeQueryTool(
                chroma_client=self.chroma_client, 
                llm=self.llm,
                query_expander=self.query_expander
            ),
        ]
        logger.info(f"Initialized {len(self.tools)} tools")
    
    def setup_agent(self):
        """Initialize the LangChain agent"""
        system_message = SystemMessage(content="""You are a European bird expert and guide. Your role is to help users with birdwatching questions.

**CRITICAL Tool Selection Rules:**
1. Use `bird_query` ONLY for questions about SPECIFIC BIRD SPECIES (e.g., "tell me about robins", "what does a sparrow look like")
2. Use `youtube_query` for ALL OTHER birdwatching topics including:
   - Communities, groups, clubs ("birding community", "birdwatching groups")  
   - Tips, techniques, guides ("birdwatching tips", "how to identify birds")
   - Equipment, habitats, behavior ("binoculars", "bird behavior")
   - General questions ("getting started with birdwatching")

**Response Format:**
- For **bird queries**: Provide a detailed description of the bird and include the image URL at the end as: `Image: [URL]`
- For **YouTube queries**: Extract the `summary` from the tool output and present it conversationally. Then add: "I found this from {video_count} videos. You can watch more at: [first URL]"

**Examples:**
User: "birding community" → Use youtube_query (NOT bird_query)
User: "robin bird" → Use bird_query  
User: "birdwatching tips" → Use youtube_query
User: "what is a sparrow" → Use bird_query""")
        
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
            transcribed_text = self.audio_processor.transcribe_audio(audio_file_path)
            
            if not transcribed_text:
                return {
                    "answer": "Sorry, I couldn't understand the audio. Please try again with clearer speech.",
                    "images": [],
                    "transcription": None,
                    "error": True
                }
            
            logger.info(f"Transcribed audio: {transcribed_text}")
            
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