# File: src/agents/bird_agent.py

import json
import logging
import re
from typing import Dict, Any, List, Optional
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
        "Search for comprehensive information about a specific bird species or a description of a bird. The database also includes audio and images."
        "This tool handles audio and image URLs internally and does not expose them to the assistant. Always expect images to be displayed automatically"
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
                
                # Store metadata separately (LLM never sees this)
                bird_metadata = {
                    "species": species,
                    "image_url": metadata.get("thumbnail", ""),
                    "audio_url": metadata.get("audio_url", ""),
                    "page_url": metadata.get("url", "")
                }
                self._stored_metadata.append(bird_metadata)
                
                # Extract clean description for LLM
                description_part = doc.split("DETAILED_INFO:")[1].strip() if "DETAILED_INFO:" in doc else doc
                clean_description = strip_markdown_links(description_part)[:1000].strip()
                
                clean_descriptions.append({
                    "species": species,
                    "description": clean_description
                })

            if not clean_descriptions:
                return f"No detailed information found for '{cleaned_query}'."
            
            # Return ONLY clean text to LLM (no URLs, no metadata)
            if len(clean_descriptions) == 1:
                bird = clean_descriptions[0]
                return f"Found {bird['species']}: {bird['description']}"
            else:
                # Multiple results
                result_text = f"Found {len(clean_descriptions)} birds:\n"
                for bird in clean_descriptions:
                    result_text += f"- {bird['species']}: {bird['description'][:200]}...\n"
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
        
        # Initialize tools with references we can access later
        self.bird_tool = BirdQueryTool(chroma_client=self.chroma_client)
        self.youtube_tool = YouTubeQueryTool(chroma_client=self.chroma_client)
        self.tools = [self.bird_tool, self.youtube_tool]

        agent_system_prompt = """
You are a helpful and friendly AI assistant who is an expert on all things birds. You are a conversational and engaging parrot.

You have access to the following tools: {tools}

**Your only purpose is to answer questions about birds.**

**Instructions:**
- **Use your tools when the user's query is directly about a bird species or bird-related topics.**
- For any other conversation, including greetings, small talk, or off-topic questions, respond directly and conversationally without using any tools, but try to get back on the topic of birds.
- When using the bird_query tool, use the information it provides to give a friendly, knowledgeable response about the bird.
- If the tool indicates no information was found, you can offer general knowledge but state it's from your general knowledge.
- For birdwatching advice, use the youtube_query tool.
- Always provide your final answer in a friendly, conversational tone. Do not mention tools or databases unless a search explicitly failed.
- Focus on being helpful and engaging while sharing bird knowledge.
        """.format(tools=[f"{tool.name}: {tool.description}" for tool in self.tools])

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=False,  # We don't need these anymore
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