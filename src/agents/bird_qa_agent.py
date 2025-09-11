
import json
import logging
import os
from typing import List, Dict, Any
import re
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


def strip_markdown_links(text: str) -> str:
    """Strip Markdown image and link syntax from text."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'Here is an image of .*?:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'You can listen to its call\s*[^.]*\.', '', text, flags=re.IGNORECASE)
    text = ' '.join(text.split())
    return text

class BirdQueryTool(BaseTool):
    name: str = "bird_query"
    description: str = """Search for comprehensive information about a specific European bird species by name.

    USE THIS TOOL WHEN:
    - User asks about a specific bird's appearance, call, song, habitat, or other facts.
    - User wants to know "what does a [bird species] sound like?", "what does a [bird species] look like?", or "tell me about a [bird species]".
    - User provides a bird name and asks for any kind of detail about it.

    INPUT: The bird species name (e.g., "european robin", "barn owl", "blue tit")
    OUTPUT: A JSON object containing the bird's species name, a detailed description, a link to an image, and a link to an audio recording.
    """
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
                logger.info(f"Raw document content: {doc[:200]}")
                metadata = results['metadatas'][0][0]

                species = metadata.get('species', 'Unknown')
                image_url = metadata.get('thumbnail', '')
                audio_url = metadata.get('audio_url', '')

                # Ensure the description does not contain any Markdown links
                description = doc[:1000].replace('![](', '').replace(')','') if doc else "No description available."

                result_data = {
                    "species": species,
                    "description": description,
                    "image_url": image_url if image_url else "",
                    "audio_url": audio_url if audio_url else ""
                }

                formatted_output = json.dumps(result_data)
                logger.info(f"BirdQueryTool returning: {formatted_output}")
                return formatted_output
            else:
                result_data = {
                    "species": "Unknown",
                    "description": f"I couldn't find specific information about '{query}'. Try asking about a common European bird.",
                    "image_url": "",
                    "audio_url": ""
                }
                return json.dumps(result_data)

        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            result_data = {
                "species": "Error",
                "description": f"I encountered an error searching for '{query}'. Please try rephrasing your question.",
                "image_url": "",
                "audio_url": ""
            }
            return json.dumps(result_data)

        
        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            result_data = {
                "species": "Error",
                "description": f"I encountered an error searching for '{query}'. Please try rephrasing your question.",
                "image_url": "",
                "audio_url": ""
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
    def __init__(self):
        self.chroma_client = ChromaClient()
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
        # Keep the system message simple, the logic for JSON output is now in `ask`.
        system_message = SystemMessage(content="""You are a helpful European birdwatching expert assistant.
        You have access to tools that can provide information about specific bird species and general birdwatching tips.
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
            # 1. Invoke the agent to get a conversational response.
            # The agent will get the data from the tool and then format it.
            response = self.agent.invoke({"input": question})
            raw_output = response['output']

            # 2. Check the raw output for URLs.
            # This is the key step: we extract the URLs first.
            image_url_match = re.search(r'!\[.*?\]\((.*?)\)', raw_output)
            audio_url_match = re.search(r'\[.*?\]\((.*?)\)', raw_output)
            
            image_url = image_url_match.group(1) if image_url_match else ""
            audio_url = audio_url_match.group(1) if audio_url_match else ""

            # 3. Clean the text from Markdown and common phrases.
            # This makes sure the final 'answer' is chatty and clean.
            clean_text = strip_markdown_links(raw_output)

            # 4. Return the final, structured JSON.
            return {
                "answer": clean_text,
                "image_url": image_url,
                "audio_url": audio_url,
                "error": False
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": "I encountered a general error. Please try rephrasing your question.",
                "image_url": "",
                "audio_url": "",
                "error": True
            }