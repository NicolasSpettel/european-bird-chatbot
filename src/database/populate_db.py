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
        return [], []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    return all_data

def process_wikipedia_data(bird_data, doc_id):
    """Process Wikipedia bird data"""
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
    if not video_data.get('transcript') or len(video_data['transcript']) < 100:
        return None, None, None
    
    doc_text = f"{video_data['title']}\n\n"
    if video_data.get('author'):
        doc_text += f"Author: {video_data['author']}\n\n"
    doc_text += f"Transcript: {video_data['transcript']}"
    
    metadata = {
        'title': video_data['title'],
        'author': video_data.get('author', ''),
        'category': video_data.get('category', 'birdwatching'),
        'source': 'YouTube',
        'url': video_data['url'],
        'video_id': video_data['video_id'],
        'type': 'youtube_content'
    }
    
    return doc_text, metadata, f"youtube_{doc_id:03d}"

def populate_chromadb():
    """Populate ChromaDB with combined data"""
    all_data = load_combined_data()
    
    if not all_data:
        logger.error("No data to process")
        return False
    
    chroma_client = ChromaClient()
    
    # Separate data by type
    bird_docs = []
    bird_metadata = []
    bird_ids = []
    
    youtube_docs = []
    youtube_metadata = []
    youtube_ids = []
    
    bird_counter = 0
    youtube_counter = 0
    
    for item in all_data:
        if item.get('type') == 'wikipedia_bird':
            doc_text, metadata, doc_id = process_wikipedia_data(item, bird_counter)
            if doc_text:
                bird_docs.append(doc_text)
                bird_metadata.append(metadata)
                bird_ids.append(doc_id)
                bird_counter += 1
                
        elif item.get('type') == 'youtube_video':
            doc_text, metadata, doc_id = process_youtube_data(item, youtube_counter)
            if doc_text:
                youtube_docs.append(doc_text)
                youtube_metadata.append(metadata)
                youtube_ids.append(doc_id)
                youtube_counter += 1
    
    try:
        # Delete old collections
        chroma_client.delete_collection("bird_descriptions")
        chroma_client.delete_collection("youtube_content")
        
        # Add bird data
        if bird_docs:
            chroma_client.add_data(
                collection_name="bird_descriptions",
                documents=bird_docs,
                metadata=bird_metadata,
                ids=bird_ids
            )
            if test_results and test_results.get("documents") and test_results["documents"][0]:
                logger.info(f"Added {len(bird_docs)} bird descriptions")
        
        # Add YouTube data
        if youtube_docs:
            chroma_client.add_data(
                collection_name="youtube_content",
                documents=youtube_docs,
                metadata=youtube_metadata,
                ids=youtube_ids
            )
            if test_results and test_results.get("documents") and test_results["documents"][0]:
                logger.info(f"Added {len(youtube_docs)} YouTube transcripts")
        
        # Verify with test searches
        if bird_docs:
            test_results = chroma_client.search(
                collection_name="bird_descriptions",
                query="red breast small bird",
                n_results=3
            )
            if test_results and test_results['documents'][0]:
                logger.info("Bird database verification successful")
        
        if youtube_docs:
            test_results = chroma_client.search(
                collection_name="youtube_content",
                query="beginner birdwatching tips",
                n_results=2
            )
            if test_results and test_results['documents'][0]:
                logger.info("YouTube database verification successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate database: {e}")
        return False

if __name__ == "__main__":
    success = populate_chromadb()
    
    if success:
        logger.info("Database population complete")
    else:
        logger.error("Database population failed")