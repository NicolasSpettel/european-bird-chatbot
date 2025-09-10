import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import json
from pathlib import Path
from src.database.chroma_client import ChromaClient
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_combined_data():
    """Load combined Wikipedia and YouTube data"""
    data_file = Path("data/raw/combined_data.json")
    if not data_file.exists():
        logger.error("Combined data file not found. Run data collection first.")
        return []
    with open(data_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    return all_data

def process_wikipedia_data(bird_data, doc_id):
    """Process Wikipedia bird data"""
    logger.info(f"Thumbnail type: {type(bird_data.get('thumbnail'))}, value: {bird_data.get('thumbnail')}")

    if not bird_data.get('extract') or len(bird_data['extract']) < 100:
        return None, None, None
    doc_text = f"{bird_data['title']}\n\n"
    if bird_data.get('description'):
        doc_text += f"Description: {bird_data['description']}\n\n"
    doc_text += f"Information: {bird_data['extract']}"
    metadata = {
        'species': bird_data['title'],
        'family': "Unknown",
        'region': 'Europe',
        'source': 'Wikipedia',
        'url': bird_data.get('page_url', ''),
        'thumbnail': bird_data.get('thumbnail', ''),
        'search_name': bird_data.get('original_search', bird_data['title']),
        'type': 'bird_description'
    }
    return doc_text, metadata, f"bird_{doc_id:03d}"

def process_youtube_data(video_data, doc_id):
    """Process YouTube video data"""
    
    # Check if 'content' and 'metadata' fields exist at the top level
    if not video_data.get('content') or not video_data.get('metadata'):
        return None, None, None
    
    # Use the 'content' field as the main document text.
    # It already contains the title, author, and transcript.
    doc_text = video_data['content']

    # Use the 'metadata' field directly as the metadata dictionary.
    metadata = video_data['metadata']

    # The length check can be done on the full document text.
    if len(doc_text) < 100:
        return None, None, None
    
    # Use the 'id' field from the JSON or create a fallback.
    doc_id_final = video_data.get('id', f"youtube_{doc_id:03d}")

    # Return the correctly extracted document, metadata, and ID
    return doc_text, metadata, doc_id_final

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
        # CORRECT THE TYPE HERE: Change 'youtube_video' to 'youtube_chunk'
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
        
        # You need to add a verification for the youtube collection as well
        if collections_data["birds"]["docs"]:
            test_results_birds = chroma_client.search(
                collection_name="birds",
                query="red breast small bird",
                n_results=3
            )
            if test_results_birds and test_results_birds.get('documents', [[]])[0]:
                logger.info("Bird database verification successful")
        
        if collections_data["youtube"]["docs"]:
            test_results_youtube = chroma_client.search(
                collection_name="youtube",
                query="beginner birdwatching tips",
                n_results=3
            )
            if test_results_youtube and test_results_youtube.get('documents', [[]])[0]:
                logger.info("YouTube database verification successful")

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