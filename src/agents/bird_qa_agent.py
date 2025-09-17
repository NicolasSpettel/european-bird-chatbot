# File: src/agents/bird_agent.py

import json
import logging
import re
from typing import Dict, Any, List
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from src.database.chroma_client import ChromaClient
from src.config import Config
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 
logger = logging.getLogger(__name__)

def strip_markdown_links(text):
    """Removes Markdown image links from a string."""
    return re.sub(r'!\[.*?\]\((.*?)\)', '', text)

class BirdQueryTool(BaseTool):
    """Tool for querying bird species information with internal metadata separation."""
    name: str = "bird_query"
    description: str = (
        "Search for comprehensive information about a specific bird species or a description of a bird. "
        "Input: The bird species name (e.g., 'european robin') OR a descriptive phrase (e.g., 'a small bird with a red chest'). "
        "Output: Clean species information for conversation."
    )
    chroma_client: ChromaClient
    _stored_metadata: List[Dict[str, str]] = []
    
    def __init__(self, **data):
        super().__init__(**data)
        self._stored_metadata = []
    
    def _run(self, query: str) -> str:
        """Execute bird query and separate metadata from conversational content."""
        try:
            cleaned_query = query.strip().strip('`').strip()
            logger.info(f"Searching for bird with cleaned query: '{cleaned_query}'")
            
            results = self.chroma_client.search(
                collection_name="birds",
                query=cleaned_query,
                n_results=2
            )
            
            self._stored_metadata = []
            
            if not results.get("documents") or not results["documents"][0]:
                logger.info(f"No results found for query: '{cleaned_query}'")
                return f"No detailed information found for '{cleaned_query}'."

            clean_descriptions = []
            
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                
                species = metadata.get("title", "Unknown")
                
                if species == "Unknown":
                    logger.warning(f"Found 'Unknown' species for query '{cleaned_query}'. Treating as no match.")
                    continue
                
                bird_metadata = {
                    "species": species,
                    "image_url": metadata.get("thumbnail", ""),
                    "audio_url": metadata.get("audio_url", ""),
                    "page_url": metadata.get("url", "")
                }
                self._stored_metadata.append(bird_metadata)
                
                description_part = doc.split("DETAILED_INFO:")[1].strip() if "DETAILED_INFO:" in doc else doc
                clean_description = strip_markdown_links(description_part)[:1000].strip()
                
                clean_descriptions.append({
                    "species": species,
                    "description": clean_description
                })

            if not clean_descriptions:
                return f"No detailed information found for '{cleaned_query}'."
            
            # ✨ New logic to return a more definitive, non-negotiable tool output
            bird = clean_descriptions[0]
            if len(clean_descriptions) == 1:
                return f"The database search found information for the following bird: {bird['species']}. Description: {bird['description']}"
            else:
                 # If multiple results, provide a clear ranking or summary
                result_text = f"The database search found a few birds that match. The top result is {bird['species']}. Description: {bird['description'][:200]}... Other birds found are:\n"
                for other_bird in clean_descriptions[1:]:
                     result_text += f"- {other_bird['species']}: {other_bird['description'][:100]}...\n"
                return result_text
        
        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            self._stored_metadata = []
            return f"An error occurred while searching for '{cleaned_query}'."
        
    def get_stored_metadata(self) -> List[Dict[str, str]]:
        """Get metadata without exposing it to LLM."""
        return self._stored_metadata.copy()
    
    def clear_stored_metadata(self):
        """Clear stored metadata."""
        self._stored_metadata = []

class YouTubeQueryTool(BaseTool):
    """Tool for querying YouTube for birdwatching advice."""
    name: str = "youtube_query"
    description: str = (
        "Find general advice and tips related to birdwatching, such as "
        "how to start birdwatching, recommended gear, or where to find birds. "
        "Use this for questions like 'How can I get into birding?', 'What equipment do I need?' "
        "Input: a concise query (e.g., 'birding tips for beginners'). "
        "Output: A summary of relevant expert advice."
    )
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        """Execute the YouTube query and return a summary."""
        try:
            logger.info(f"Searching YouTube for: {query}")
            results = self.chroma_client.search(
                collection_name="youtube",
                query=query,
                n_results=2
            )
            if not results or not results["documents"] or not results["documents"][0]:
                return "No expert advice found on this topic."

            summaries = []
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                summaries.append(f"From '{metadata.get('title', 'Expert Video')}': {doc[:800].strip()}...")

            return "\n\n".join(summaries)

        except Exception as e:
            logger.error(f"YouTube query failed: {e}")
            return f"Error searching for birdwatching advice about '{query}'."

class BirdQAAgent:
    """Agent for answering bird-related questions using tools and LLM."""

    def __init__(self):
        self.chroma_client = ChromaClient()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.bird_tool = BirdQueryTool(chroma_client=self.chroma_client)
        self.youtube_tool = YouTubeQueryTool(chroma_client=self.chroma_client)
        self.tools = [self.bird_tool, self.youtube_tool]

        agent_system_prompt = """
You are an expert AI assistant (embodied as a smart parrot) specializing in all things related to European birds. Your knowledge and purpose are strictly limited to providing information on this topic.

Primary Directives:

For ANY bird-related query, regardless of how general it is, you MUST first use your bird_query tool to search for information. This is the foundational step for every response.

Analyze the search results from the bird_query tool.

If the tool provides information about a specific bird, construct a comprehensive and helpful response based on those results.

If the results are not a perfect match or don't provide a complete answer (e.g., a general question like "do birds get hungry?"), still acknowledge the search. You should then use your broader knowledge to provide a general answer to the user's question, mentioning that the tool found related but not directly relevant information. The goal is to always use the tool and then reason about its output.

If the query is NOT about birds or any bird-related topic (e.g., "What is the weather?"), you must politely decline to answer. Your response must be: "I'm sorry, but my knowledge is focused on European birds. Please ask me a question about birds!" This is your final, mandatory fallback for truly out-of-scope queries.
"""

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            # Change AgentType to use OpenAI's native tool calling
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=False,
            # The tool-calling agent is more robust and less prone to parsing errors,
            # so `handle_parsing_errors=True` is not strictly needed but can be kept
            # for an extra layer of safety.
            handle_parsing_errors=True,
            # Remove `agent_kwargs` with `format_instructions`
            # as the tool-calling agent doesn't need them.
            agent_kwargs={"system_message": agent_system_prompt}
        )

    def clear_memory(self):
        """Clear the agent's conversation memory."""
        try:
            if hasattr(self, "memory") and self.memory:
                self.memory.clear()
                logger.info("Memory cleared.")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    def ask(self, user_query: str) -> Dict[str, Any]:
        """Ask the agent a question and get a structured response."""
        try:
            self.bird_tool.clear_stored_metadata()
            
            response = self.agent_executor.invoke({"input": user_query})
            final_response = response.get("output", "I'm sorry, I couldn't find an answer.")
            
            found_birds = self.bird_tool.get_stored_metadata()
            
            birds_for_frontend = []
            for bird_meta in found_birds:
                birds_for_frontend.append({
                    "species": bird_meta["species"],
                    "image_url": bird_meta["image_url"],
                    "audio_url": bird_meta["audio_url"]
                })

            return {
                "response": strip_markdown_links(final_response),
                "birds": birds_for_frontend,
                "error": False,
            }
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "response": "I encountered an error. Please try again.",
                "birds": [],
                "error": True,
                "message": str(e)
            }

    def process_audio_bytes(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Process audio bytes using OpenAI Whisper and return the transcription."""
        try:
            logger.info(f"Processing audio file: {filename}")
            temp_path = "temp_audio.mp3"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return {
                "transcription": transcription.text,
                "error": False
            }
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "transcription": "",
                "error": True,
                "message": str(e)
            }