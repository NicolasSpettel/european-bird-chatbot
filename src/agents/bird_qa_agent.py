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
            
            # **MODIFIED:** Handle cases where no results are found.
            if not results or not results.get("documents") or not results["documents"][0]:
                return json.dumps([{"species": "No Match Found", "description": f"No detailed information found for a query about '{query}'.", "image_url": "", "audio_url": ""}])

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
            # **MODIFIED:** Return a specific error object that the LLM can interpret.
            return json.dumps([{
                "species": "Error",
                "description": f"An error occurred while searching for '{query}'.",
                "image_url": "",
                "audio_url": "",
            }])


class YouTubeQueryTool(BaseTool):
    """Tool for querying YouTube for birdwatching advice."""
    name: str = "youtube_query"
    description: str = (
    "This tool is for finding general advice and tips related to birdwatching, such as "
    "how to start birdwatching, recommended gear, or where to find birds. "
    "Use this for questions like 'How can I get into birding?', 'What equipment do I need?', or 'How can I get my friend interested in birding?' "
    "Input: a concise query (e.g., 'birding tips for beginners', 'how to get a friend into birding'). "
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

        # New: Define the conversational prompt
        agent_system_prompt = agent_system_prompt = agent_system_prompt = agent_system_prompt = """
            You are a helpful and friendly AI assistant who is an expert on all things birds. You are a conversational and engaging parrot.

            You have access to the following tools: {tools}

            **Your only purpose is to answer questions about birds.**

            **Instructions:**
            - **Only use your tools when the user's query is directly about birds or bird-related topics.**
            - For any other conversation, including greetings, small talk, or off-topic questions, respond directly and conversationally without using any tools.
            - When using the `bird_query` tool, carefully analyze the results. If a bird is found, provide a friendly and concise answer using the provided information. If multiple birds are found, you may mention the most relevant one and suggest the user can ask for more details on the others.
            - If a tool search for a bird yields "No Match Found," inform the user and ask if they would like to know about a different bird or a more general topic.
            - Always provide your final answer in a friendly, conversational tone. Do not mention any tools or databases in your final response.
            """.format(tools=[tool.name + ": " + tool.description for tool in self.tools])

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={"system_message": agent_system_prompt}
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
        try:
            # Let the agent's internal logic handle everything
            response = self.agent_executor.invoke({"input": user_query})
            final_answer = response.get("output", "I'm sorry, I couldn't find an answer.")
            # Extract data from intermediate steps if a tool was used
            image_url = ""
            audio_url = ""
            species = ""
            intermediate_steps = response.get("intermediate_steps", [])

            for action, observation in intermediate_steps:
                if action.tool == "bird_query":
                    try:
                        bird_data_list = json.loads(observation)
                        if bird_data_list and bird_data_list[0].get("species") != "No Match Found":
                            selected_bird = bird_data_list[0]
                            image_url = selected_bird.get("image_url", "")
                            audio_url = selected_bird.get("audio_url", "")
                            species = selected_bird.get("species", "")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse bird data.")
            
            # Return a single, unified output
            return {
                "answer": strip_markdown_links(final_answer),
                "species": species,
                "image_url": image_url,
                "audio_url": audio_url,
                "error": False,
            }
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": "I encountered an error. Please try again.",
                "species": "",
                "image_url": "",
                "audio_url": "",
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