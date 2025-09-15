# File: src/agents/bird_agent.py

import json
import logging
import re
from typing import Dict, Any, List
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.database.chroma_client import ChromaClient
from src.config import Config
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

def strip_markdown_links(text):
    """Removes Markdown image links from a string."""
    return re.sub(r'!\[.*?\]\((.*?)\)', '', text)

class BirdQueryTool(BaseTool):
    """Tool for querying bird species information from the Chroma database. Handles both name and descriptive queries."""
    name: str = "bird_query"
    description: str = (
        "Search for comprehensive information about a specific bird species or a description of a bird. "
        "Input: The bird species name (e.g., 'european robin') OR a descriptive phrase (e.g., 'a small bird with a red chest'). "
        "Output: JSON list with keys: 'species', 'description', 'image_url', 'audio_url'."
    )
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        """Execute the bird query and return structured results."""
        try:
            logger.info(f"Searching for bird with query: '{query}'")
            
            # Use the descriptive search to find top 3 relevant birds
            results = self.chroma_client.search(
                collection_name="birds",
                query=query,
                n_results=3
            )
            
            if not results or not results["documents"] or not results["documents"][0]:
                return json.dumps([{"species": "Unknown", "description": f"No information found for '{query}'.", "image_url": "", "audio_url": ""}])

            output_results = []
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                
                species = metadata.get("species", "Unknown")
                image_url = metadata.get("thumbnail", "")
                audio_url = metadata.get("audio_url", "")
                
                # Truncate the description for cleaner output
                description_part = doc.split("DETAILED_INFO:")[1].strip() if "DETAILED_INFO:" in doc else doc
                description = description_part[:500].replace("![](", "").replace(")", "")
                
                result = {
                    "species": species,
                    "description": description,
                    "image_url": image_url,
                    "audio_url": audio_url,
                }
                output_results.append(result)

            logger.info(f"Returning bird data: {output_results}")
            return json.dumps(output_results)

        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            return json.dumps([{
                "species": "Error",
                "description": f"Error searching for '{query}'.",
                "image_url": "",
                "audio_url": "",
            }])


class YouTubeQueryTool(BaseTool):
    """Tool for querying YouTube for birdwatching advice."""
    name: str = "youtube_query"
    description: str = (
        "Search YouTube for educational content on birdwatching advice and techniques. "
        "Input: User query (string). Output: Plain text summary."
    )
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        """Execute the YouTube query and return a summary."""
        try:
            logger.info(f"Searching YouTube for: {query}")
            results = self.chroma_client.search(
                collection_name="youtube",
                query=query,
                n_results=3
            )
            if not results or not results["documents"] or not results["documents"][0]:
                return "No expert advice found on this topic."

            # Summarize the top 3 transcripts for the LLM
            summaries = []
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                summaries.append(f"Video Title: {metadata.get('title', 'N/A')}\nTranscript excerpt: {doc[:300].strip()}...\nURL: {metadata.get('url', 'N/A')}")

            return "\n\n---\n\n".join(summaries)

        except Exception as e:
            logger.error(f"YouTube query failed: {e}")
            return f"Error searching for '{query}'."

class BirdQAAgent:
    """Agent for answering bird-related questions using tools and LLM."""

    def __init__(self):
        self.chroma_client = ChromaClient()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.tools = [
            BirdQueryTool(chroma_client=self.chroma_client),
            YouTubeQueryTool(chroma_client=self.chroma_client)
        ]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True
        )
        self.last_image_url = ""
        self.last_audio_url = ""

    def clear_memory(self):
        """Clear the agent's conversation memory."""
        try:
            if hasattr(self, "memory") and self.memory:
                self.memory.clear()
                logger.info("Memory cleared.")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    def _validate_species_match(self, user_query: str, retrieved_species: str) -> bool:
        """Check if the retrieved species matches the user's query."""
        user_query_normalized = user_query.lower().strip()
        species_normalized = retrieved_species.lower().strip()

        validation_prompt = PromptTemplate(
            input_variables=["query", "species"],
            template=(
                "Is '{species}' a good match for the user's query about '{query}'? "
                "Answer 'YES' if it's a direct or very close match, and 'NO' if it's not. "
                "Only output 'YES' or 'NO'."
            )
        )
        validation_chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        validation_result = validation_chain.run(query=user_query_normalized, species=species_normalized)

        return "YES" in validation_result.upper()

    def ask(self, user_query: str) -> Dict[str, Any]:
        """Answer a user's question about birds."""
        try:
            response = self.agent_executor.invoke({"input": user_query})
            raw_output = response["output"]
            intermediate_steps = response["intermediate_steps"]

            bird_data = {}
            for action, observation in intermediate_steps:
                if action.tool == "bird_query":
                    try:
                        # The tool now returns a list, so load it as such
                        data_list = json.loads(observation)
                        if data_list:
                            bird_data = data_list[0]  # Get the top result for display
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse bird data: {observation}")

            species = bird_data.get("species", "")
            image_url = bird_data.get("image_url", "")
            audio_url = bird_data.get("audio_url", "")
            description = bird_data.get("description", "")
            
            # New: Handle descriptive searches by formatting the LLM's raw output
            if not species and "No information found" not in description:
                # This path is for when the LLM uses the tool for a descriptive query
                # but the raw output from the tool is complex.
                # We'll re-prompt the LLM to format the response.
                summary_prompt = PromptTemplate(
                    input_variables=["user_query", "raw_output"],
                    template=(
                        "The user asked: '{user_query}'. "
                        "The following information was found: {raw_output}. "
                        "Synthesize this information into a clear and helpful response. "
                        "Do not mention any tools or databases. "
                        "If multiple birds are returned, mention a few of the top matches and describe them briefly. "
                        "Prioritize the most relevant bird for the user's query."
                    )
                )
                summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
                answer_text = summary_chain.run(user_query=user_query, raw_output=raw_output)
            elif species:
                # This path is for specific bird queries that get a clear result.
                # It includes the validation logic.
                is_match = self._validate_species_match(user_query, species)
                if not is_match:
                    return {
                        "answer": f"I couldn't find information about '{user_query}'. The database returned '{species}', which doesn't seem to be a direct match. Did you mean another bird?",
                        "species": "",
                        "image_url": self.last_image_url,
                        "audio_url": self.last_audio_url,
                        "error": False,
                    }
                answer_prompt = PromptTemplate(
                    input_variables=["user_query", "species", "description"],
                    template=(
                        "You are a birdwatching expert. The user asked: '{user_query}'. "
                        "Here's what you know about the {species}: {description}. "
                        "Provide a friendly, natural, and expert answer. Do not mention tools or databases." \
                        "Keep the answer concise and engaging."
                    )
                )
                answer_chain = LLMChain(llm=self.llm, prompt=answer_prompt)
                answer_text = answer_chain.run(user_query=user_query, species=species, description=description)
                
                # Update last valid image and audio URLs if new data is available
                if image_url: self.last_image_url = image_url
                if audio_url: self.last_audio_url = audio_url
            else:
                # Generic fallback for no results
                answer_text = "I couldn't find relevant information. Could you rephrase or ask about a specific bird?"
                
            return {
                "answer": strip_markdown_links(answer_text).strip(),
                "species": species,
                "image_url": self.last_image_url,
                "audio_url": self.last_audio_url,
                "error": False,
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": "I encountered an error. Please try again.",
                "species": "",
                "image_url": self.last_image_url,
                "audio_url": self.last_audio_url,
                "error": True,
            }

    def process_audio_bytes(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Process audio bytes using OpenAI Whisper and return the transcription.
        Args:
            audio_bytes: Raw audio bytes.
            filename: Name of the audio file (for logging).
        Returns:
            Dictionary with transcription and optional metadata.
        """
        try:
            logger.info(f"Processing audio file: {filename}")
            temp_path = "temp_audio.mp3"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Saved temporary audio file: {temp_path}")
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