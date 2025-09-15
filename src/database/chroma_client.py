# File: src/database/chroma_client.py

import sys
import os
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from src.config import Config

logger = logging.getLogger(__name__)

class ChromaClient:
    def __init__(self):

        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=Config.FINE_TUNED_MODEL_PATH
            )
            logger.info("Custom embedding function initialized with fine-tuned model.")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned embedding model: {e}")
            raise RuntimeError("Exiting due to failed model load.")

        # Initialize the ChromaDB client with persistent storage
        db_path = "./data/chroma_db"
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client initialized at {db_path}")

    def get_or_create_collection(self, collection_name: str, metadata: dict = None):
        """Get or create a collection by name, using the custom embedding function."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
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
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
            )
            return results
        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            return None