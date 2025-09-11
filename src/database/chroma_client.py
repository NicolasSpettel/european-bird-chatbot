import sys
import os
import json
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure imports work from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)

class ChromaClient:
    def __init__(self):
        self.client = None
        self.setup_client()

    def setup_client(self):
        """Initialize ChromaDB client with the custom embedding function"""
        try:
            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB client initialized with custom embedding function: BAAI/bge-large-en-v1.5")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def get_or_create_collection(self, collection_name: str, metadata: dict = None):
        """Get or create a collection by name, using the custom embedding function."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,  # Explicitly pass the custom embedding function
                metadata=metadata or {"description": f"{collection_name} collection"},
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection '{collection_name}': {e}")
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
            collection = self.get_or_create_collection(collection_name)
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
            return results
        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            return None

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
            meta = {
                'species': bird['title'],
                'family': bird.get('family', 'Unknown'),
                'region': 'Europe',
                'source': 'Wikipedia',
                'url': bird.get('page_url', ''),
                'thumbnail': bird.get('thumbnail', {}).get('source', '') if bird.get('thumbnail') else '',
                'search_name': bird.get('search_name', bird['title'])
            }
            metadata.append(meta)
            ids.append(f"bird_{i:03d}")
    return documents, metadata, ids

def populate_chromadb():
    """Populate ChromaDB with combined data"""
    all_data = load_combined_data()
    if not all_data:
        logger.error("No data to process")
        return False

    chroma_client = ChromaClient()

    # Use a single dictionary to hold documents, metadata, and ids for each collection
    collections_data = {
        "birds": {"docs": [], "metadatas": [], "ids": []},
        "youtube": {"docs": [], "metadatas": [], "ids": []}
    }
    
    # Process data and populate the dictionary
    bird_counter, youtube_counter = 0, 0
    for item in all_data:
        if item.get('type') == 'wikipedia_bird':
            doc_text, metadata, doc_id = process_wikipedia_data(item, bird_counter)
            if doc_text:
                collections_data["birds"]["docs"].append(doc_text)
                collections_data["birds"]["metadatas"].append(metadata)
                collections_data["birds"]["ids"].append(doc_id)
                bird_counter += 1
        elif item.get('type') == 'youtube_chunk':
            doc_text, metadata, doc_id = process_youtube_data(item, youtube_counter)
            if doc_text:
                collections_data["youtube"]["docs"].append(doc_text)
                collections_data["youtube"]["metadatas"].append(metadata)
                collections_data["youtube"]["ids"].append(doc_id)
                youtube_counter += 1
    
    try:
        # Loop through the dictionary to add data to each collection
        for name, data in collections_data.items():
            if data["docs"]:
                chroma_client.add_data(
                    collection_name=name,
                    documents=data["docs"],
                    metadata=data["metadatas"],
                    ids=data["ids"]
                )
                logger.info(f"Added {len(data['docs'])} documents to '{name}'")
        
        # Verification can remain the same
        # ... (your existing verification logic here) ...
        
        return True
    except Exception as e:
        logger.error(f"Failed to populate database: {e}")
        return False

if __name__ == "__main__":
    success = populate_chromadb()
    if success:
        logger.info("Database population complete. Ready for agent creation.")
    else:
        logger.error("Database population failed. Check errors above.")
