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
    # Handle missing 'extract' and short extracts
    extract = bird_data.get('extract')
    if not extract or len(extract) < 100:
        return None, None, None

    # Ensure all values are strings for ChromaDB metadata
    metadata = {
        'species': str(bird_data.get('title', '')),
        'family': "Unknown",
        'region': 'Europe',
        'source': 'Wikipedia',
        'url': str(bird_data.get('page_url', '')),
        'thumbnail': str(bird_data.get('thumbnail', '')),
        'search_name': str(bird_data.get('original_search', bird_data.get('title', ''))),
        'audio_url': str(bird_data.get('audio_url', '')),
        'type': 'bird_description'
    }

    doc_text = (f"{metadata['species']}\n\n"
                f"Description: {bird_data.get('description', '')}\n\n"
                f"Information: {extract}")

    return doc_text, metadata, f"bird_{doc_id:03d}"

def process_youtube_data(video_data, doc_id):
    """Process YouTube video data"""
    if not video_data.get('content') or not video_data.get('metadata'):
        return None, None, None
    doc_text = video_data['content']
    metadata = video_data['metadata']
    if len(doc_text) < 100:
        return None, None, None
    doc_id_final = video_data.get('id', f"youtube_{doc_id:03d}")
    return doc_text, metadata, doc_id_final

def populate_chromadb():
    """Populate ChromaDB with combined data"""
    all_data = load_combined_data()
    if not all_data:
        logger.error("No data to process")
        return False
    
    chroma_client = ChromaClient()

    chroma_client.delete_collection("birds")
    chroma_client.delete_collection("youtube")
    
    bird_docs, bird_metadatas, bird_ids = [], [], []
    youtube_docs, youtube_metadatas, youtube_ids = [], [], []

    bird_counter, youtube_counter = 0, 0
    for item in all_data:
        if item.get('type') == 'wikipedia_bird':
            doc_text, metadata, doc_id = process_wikipedia_data(item, bird_counter)
            if doc_text:  # This check ensures doc_text is not None
                bird_docs.append(doc_text)
                bird_metadatas.append(metadata)
                bird_ids.append(doc_id)
                bird_counter += 1
        elif item.get('type') == 'youtube_chunk':
            doc_text, metadata, doc_id = process_youtube_data(item, youtube_counter)
            if doc_text: # This check ensures doc_text is not None
                youtube_docs.append(doc_text)
                youtube_metadatas.append(metadata)
                youtube_ids.append(doc_id)
                youtube_counter += 1
    
    try:
        if bird_docs:
            chroma_client.add_data(
                collection_name="birds",
                documents=bird_docs,
                metadata=bird_metadatas,
                ids=bird_ids
            )
            logger.info(f"Added {len(bird_docs)} documents to 'birds'")
        
        if youtube_docs:
            chroma_client.add_data(
                collection_name="youtube",
                documents=youtube_docs,
                metadata=youtube_metadatas,
                ids=youtube_ids
            )
            logger.info(f"Added {len(youtube_docs)} documents to 'youtube'")
            
        # Optional: Add your verification checks here to confirm data exists.
        
        return True
    except Exception as e:
        logger.error(f"Failed to populate database: {e}")
        return False

if __name__ == "__main__":
    success = populate_chromadb()
    if success:
        logger.info("Database population complete.")
    else:
        logger.error("Database population failed. Check errors above.")