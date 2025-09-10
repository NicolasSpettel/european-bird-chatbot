import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
from pathlib import Path
from src.database.chroma_client import ChromaClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_birds():
    """Load bird data from JSON and prepare for ChromaDB"""
    data_file = Path("data/raw/wikipedia_birds.json")
    
    if not data_file.exists():
        logger.error("Bird data file not found. Run data collection first.")
        return None, None, None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        birds_data = json.load(f)
    
    documents = []
    metadata = []
    ids = []
    
    for i, bird in enumerate(birds_data):
        if bird.get('extract') and len(bird['extract']) > 100:
            doc_text = f"{bird['title']}\n\n"
            
            if bird.get('description'):
                doc_text += f"Description: {bird['description']}\n\n"
            
            doc_text += f"Information: {bird['extract']}"
            
            documents.append(doc_text)
            
            family = "Unknown"
            description = bird.get('description', '').lower()
            if 'family' in description:
                family = "Extracted from description"
            
            # --- THIS IS THE CRITICAL FIX ---
            meta = {
                'species': bird['title'],
                'family': family,
                'region': 'Europe',
                'source': 'Wikipedia',
                'url': bird.get('page_url', ''),
                'thumbnail': bird.get('thumbnail', ''),  # Store the thumbnail URL here
                'search_name': bird.get('search_name', bird['title'])
            }
            # --- END OF FIX ---
            
            metadata.append(meta)
            
            ids.append(f"bird_{i:03d}")
    
    return documents, metadata, ids

def populate_chromadb():
    """Populate ChromaDB with bird data"""
    print("Loading bird data from Wikipedia collection...")
    
    documents, metadata, ids = load_and_process_birds()
    
    if not documents:
        print("No bird data to process")
        return False
    
    print(f"Processing {len(documents)} bird entries")
    
    # Initialize ChromaDB client
    chroma_client = ChromaClient()
    collection_name = "bird_descriptions"
    
    try:
        print(f"Deleting old collection '{collection_name}'...")
        chroma_client.delete_collection(collection_name=collection_name)

        # Add to ChromaDB
        chroma_client.add_bird_data(
            collection_name="bird_descriptions",
            documents=documents,
            metadata=metadata,
            ids=ids
        )
        
        print(f"Successfully populated ChromaDB with {len(documents)} bird species")

        # Verify by doing a search
        test_results = chroma_client.search_birds(
            collection_name="bird_descriptions",
            query="red breast small bird",
            n_results=3
        )
        
        if test_results and test_results['documents'][0]:
            print("Database verification successful")
            print(f"Sample search result: {test_results['metadatas'][0][0]['species']}")
        else:
            print("Warning: Database populated but search verification failed")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate database: {e}")
        return False

if __name__ == "__main__":
    success = populate_chromadb()import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import logging

logger = logging.getLogger(__name__)


class ChromaClient:
    def __init__(self):
        self.client = None
        self.collections = {}
        self.embedding_function = None
        self.setup_client()

    def setup_client(self):
        """Initialize ChromaDB client with OpenAI embeddings"""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small",
            )

            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False),
            )

            # Pre-create collections
            self.collections["bird_descriptions"] = self.client.get_or_create_collection(
                name="bird_descriptions",
                metadata={"description": "Bird species descriptions and information"},
                embedding_function=self.embedding_function,
            )

            self.collections["youtube_content"] = self.client.get_or_create_collection(
                name="youtube_content",
                metadata={"description": "YouTube video transcripts about birdwatching"},
                embedding_function=self.embedding_function,
            )

            logger.info("ChromaDB client initialized with OpenAI embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def delete_collection(self, collection_name: str):
        """Deletes a collection by name."""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        except Exception as e:
            logger.warning(f"Failed to delete collection '{collection_name}': {e}")

    def add_data(self, collection_name: str, documents: list, metadata: list, ids: list):
        """Add data to specified collection"""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            collection.add(documents=documents, metadatas=metadata, ids=ids)
            logger.info(f"Added {len(documents)} documents to {collection_name}")
        except Exception as e:
            logger.error(f"Failed to add data to {collection_name}: {e}")
            raise

    def search(self, collection_name: str, query: str, n_results: int = 5):
        """Search in specified collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            results = collection.query(query_texts=[query], n_results=n_results)

            docs = results.get("documents", [])
            if not docs or not docs[0]:
                logger.info(f"No results found in {collection_name} for query: {query}")
                return None

            return results
        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            return None

    
    if success:
        print("Database population complete. Ready for agent creation.")
    else:
        print("Database population failed. Check errors above.")