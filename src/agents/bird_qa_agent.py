import json
import logging
import re
from typing import Dict, Any
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

def strip_markdown_links(text: str) -> str:
    """Remove Markdown links and images from text."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'Here is an image of .*?:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'You can listen to its call\s*[^.]*\.', '', text, flags=re.IGNORECASE)
    return ' '.join(text.split())

class BirdQueryTool(BaseTool):
    """Tool for querying bird species information from the Chroma database."""
    name: str = "bird_query"
    description: str = (
        "Search for comprehensive information about a specific European bird species by name. "
        "Input: The bird species name (e.g., 'european robin'). "
        "Output: JSON with keys: species, description, image_url, audio_url."
    )
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        """Execute the bird query and return structured results."""
        try:
            logger.info(f"Searching for bird: {query}")
            results = self.chroma_client.search(
                collection_name="birds",
                query=query,
                n_results=1
            )
            if not results or not results["documents"] or not results["documents"][0]:
                return json.dumps({
                    "species": "Unknown",
                    "description": f"No information found for '{query}'.",
                    "image_url": "",
                    "audio_url": "",
                })

            doc = results["documents"][0][0]
            metadata = results["metadatas"][0][0]
            species = metadata.get("species", "Unknown")
            image_url = metadata.get("thumbnail", "")
            audio_url = metadata.get("audio_url", "")
            description = doc[:1000].replace("![](", "").replace(")", "") if doc else "No description available."

            result = {
                "species": species,
                "description": description,
                "image_url": image_url,
                "audio_url": audio_url,
            }
            logger.info(f"Returning bird data: {result}")
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            return json.dumps({
                "species": "Error",
                "description": f"Error searching for '{query}'.",
                "image_url": "",
                "audio_url": "",
            })

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

            docs = results["documents"][0]
            full_text = " ".join(docs)
            return strip_markdown_links(full_text[:1000]) if full_text else "No advice available."

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
        # Store the last valid image and audio URLs
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
            # Step 1: Run the agent
            response = self.agent_executor.invoke({"input": user_query})
            raw_output = response["output"]
            intermediate_steps = response["intermediate_steps"]

            # Step 2: Extract bird data from intermediate steps
            bird_data = {}
            for action, observation in intermediate_steps:
                if action.tool == "bird_query":
                    try:
                        bird_data = json.loads(observation)
                        break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse bird data: {observation}")

            species = bird_data.get("species", "")
            image_url = bird_data.get("image_url", "")
            audio_url = bird_data.get("audio_url", "")
            description = bird_data.get("description", "")

            # Step 3: Validate the species match
            if species and description:
                is_match = self._validate_species_match(user_query, species)
                if not is_match:
                    return {
                        "answer": f"I couldn't find information about '{user_query}'. The database returned '{species}', which doesn't seem to match. Did you mean another bird?",
                        "species": "",
                        "image_url": self.last_image_url,  # Keep the last valid image
                        "audio_url": self.last_audio_url,  # Keep the last valid audio
                        "error": False,
                    }

            # Step 4: Update last valid image and audio URLs if new data is available
            if image_url:
                self.last_image_url = image_url
            if audio_url:
                self.last_audio_url = audio_url

            # Step 5: Generate the answer
            if species and description:
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
                answer_text = answer_chain.run(
                    user_query=user_query,
                    species=species,
                    description=description
                )
            elif raw_output:
                answer_prompt = PromptTemplate(
                    input_variables=["user_query", "raw_output"],
                    template=(
                        "You are a birdwatching expert. The user asked: '{user_query}'. "
                        "Based on this information: {raw_output}. "
                        "Provide a clear, friendly answer. Do not mention tools or databases."
                    )
                )
                answer_chain = LLMChain(llm=self.llm, prompt=answer_prompt)
                answer_text = answer_chain.run(user_query=user_query, raw_output=raw_output)
            else:
                answer_text = "I couldn't find relevant information. Could you rephrase or ask about a specific bird?"

            return {
                "answer": strip_markdown_links(answer_text).strip(),
                "species": species,
                "image_url": self.last_image_url,  # Always return the last valid image
                "audio_url": self.last_audio_url,  # Always return the last valid audio
                "error": False,
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": "I encountered an error. Please try again.",
                "species": "",
                "image_url": self.last_image_url,  # Keep the last valid image on error
                "audio_url": self.last_audio_url,  # Keep the last valid audio on error
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

            # Save the audio bytes to a temporary file
            temp_path = "temp_audio.mp3"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Saved temporary audio file: {temp_path}")

            # Initialize the OpenAI client
            client = OpenAI(api_key=Config.OPENAI_API_KEY)

            # Use the new API for transcription
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Return the transcription as a response
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