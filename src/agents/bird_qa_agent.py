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
from langchain.schema.agent import AgentFinish, AgentAction

from src.database.chroma_client import ChromaClient
from src.config import Config

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
    Input: The bird species name (e.g., "european robin", "barn owl", "blue tit").
    Output: A JSON string with keys: species, description, image_url, audio_url.
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
            if results and results["documents"] and results["documents"][0]:
                doc = results["documents"][0][0]
                metadata = results["metadatas"][0][0]
                species = metadata.get("species", "Unknown")
                image_url = metadata.get("thumbnail", "")
                audio_url = metadata.get("audio_url", "")

                description = (
                    doc[:1000].replace("![](", "").replace(")", "")
                    if doc else "No description available."
                )

                result_data = {
                    "species": species,
                    "description": description,
                    "image_url": image_url,
                    "audio_url": audio_url,
                }
                formatted_output = json.dumps(result_data)
                logger.info(f"BirdQueryTool returning: {formatted_output}")
                return formatted_output
            else:
                result_data = {
                    "species": "Unknown",
                    "description": f"I couldn't find information about '{query}'.",
                    "image_url": "",
                    "audio_url": "",
                }
                return json.dumps(result_data)
        except Exception as e:
            logger.error(f"Bird query failed: {e}")
            result_data = {
                "species": "Error",
                "description": f"Error searching for '{query}'.",
                "image_url": "",
                "audio_url": "",
            }
            return json.dumps(result_data)


class YouTubeQueryTool(BaseTool):
    name: str = "youtube_query"
    description: str = """Search YouTube educational content for birdwatching advice and techniques.
    Input: User query (string).
    Output: Plain text summary.
    """
    chroma_client: ChromaClient

    def _run(self, query: str) -> str:
        try:
            logger.info(f"YouTubeQueryTool searching for: {query}")
            results = self.chroma_client.search(
                collection_name="youtube",
                query=query,
                n_results=3
            )
            if results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                full_text = " ".join(docs)
                description = strip_markdown_links(full_text[:1000]) if full_text else "No description available."
                return description
            else:
                return "I couldn't find any expert advice on that topic."
        except Exception as e:
            logger.error(f"YouTube query failed: {e}")
            return f"Error searching for '{query}'."


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
        
    def clear_memory(self):
        """Reset the conversation memory so the agent forgets past context."""
        try:
            if hasattr(self, "memory") and self.memory is not None:
                self.memory.clear()
                logger.info("Agent memory cleared.")
            else:
                logger.warning("No memory found to clear.")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    def setup_tools(self):
        self.tools = [
            BirdQueryTool(chroma_client=self.chroma_client),
            YouTubeQueryTool(chroma_client=self.chroma_client)
        ]

    def setup_agent(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Fix 1: Add `return_intermediate_steps=True` to get the tool outputs.
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True
        )

    def ask(self, question: str) -> Dict[str, Any]:
        try:
            # Step 1: Let agent run
            # Fix 2: The invoke method now returns a dictionary with 'output' and 'intermediate_steps'
            response = self.agent_executor.invoke({"input": question})
            raw_output = response["output"]
            intermediate_steps = response["intermediate_steps"]

            # Step 2: Search the intermediate steps for the tool's output
            metadata = {}
            for step in intermediate_steps:
                action, observation = step
                # Check if the tool's name is 'bird_query' and if the observation is a JSON string
                if action.tool == "bird_query":
                    try:
                        metadata = json.loads(observation)
                        break # Stop searching after finding the bird query tool output
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse JSON from tool output: {observation}")
            
            species = metadata.get("species", "")
            image_url = metadata.get("image_url", "")
            audio_url = metadata.get("audio_url", "")
            description = metadata.get("description", "")

            # Step 3: Use LLM for friendly text from description only
            if description and image_url and audio_url:
                # Prompt the LLM to write a nice summary based ONLY on the description
                prompt = PromptTemplate(
                    input_variables=["desc"],
                    template=(
                        "You are a helpful European birdwatching expert. Based on this description, "
                        "write a clear, concise, and friendly explanation about the bird. "
                        "Do not include image, audio, or markdown links. Do not reference external tools. "
                        "Focus on the content of the description itself.\n\n"
                        "{desc}"
                    )
                )
                chain = LLMChain(llm=self.llm, prompt=prompt)
                answer_text = chain.run(desc=description)
            else:
                # If there's no structured data (e.g., for YouTube queries or general chat),
                # use the raw output and clean it up.
                answer_text = strip_markdown_links(raw_output)

            return {
                "answer": answer_text.strip(),
                "species": species,
                "image_url": image_url,
                "audio_url": audio_url,
                "error": False,
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": "I encountered a general error. Please try again.",
                "species": "",
                "image_url": "",
                "audio_url": "",
                "error": True,
            }