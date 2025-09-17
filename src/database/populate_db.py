# File: src/database/populate_db.py

import os
import json
import re
from pathlib import Path
from src.database.chroma_client import ChromaClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_characteristics(text):
    """Extract bird characteristics for better descriptive search."""
    characteristics = []
    
    size_patterns = [
        (r'\b(small|tiny|little)\b', 'small'),
        (r'\b(medium|moderate)\b', 'medium'),
        (r'\b(large|big|huge)\b', 'large'),
        (r'\b(\d+)\s*cm\b', 'size'),
        (r'\b(\d+)\s*mm\b', 'size')
    ]
    
    color_patterns = [
        (r'\b(black|dark)\b', 'black'),
        (r'\b(white|pale)\b', 'white'),
        (r'\b(brown|brownish)\b', 'brown'),
        (r'\b(red|reddish|crimson)\b', 'red'),
        (r'\b(blue|bluish)\b', 'blue'),
        (r'\b(yellow|golden|buff)\b', 'yellow'),
        (r'\b(green|olive)\b', 'green'),
        (r'\b(grey|gray|ashy)\b', 'grey'),
        (r'\b(orange|rufous)\b', 'orange'),
        (r'\b(spotted|speckled|mottled)\b', 'spotted'),
        (r'\b(striped|streaked|barred)\b', 'striped')
    ]
    
    body_patterns = [
        (r'\b(long|short|pointed|curved|thick|thin|slender)\s+(bill|beak)\b', 'beak'),
        (r'\b(long|short|pointed|square|forked)\s+tail\b', 'tail'),
        (r'\b(long|short|stilt|webbed)\s+legs\b', 'legs'),
        (r'\b(broad|narrow|pointed|rounded)\s+wings\b', 'wings'),
        (r'\b(crested|crowned)\b', 'crest'),
        (r'\b(breast|chest|belly|throat|head|back|rump)\b', 'bodypart')
    ]
    
    habitat_patterns = [
        (r'\b(woodland|forest|trees)\b', 'woodland'),
        (r'\b(water|wetland|marsh|pond|lake|river)\b', 'water'),
        (r'\b(garden|park|urban)\b', 'urban'),
        (r'\b(grassland|field|meadow)\b', 'grassland'),
        (r'\b(coast|shore|beach|cliff)\b', 'coastal'),
        (r'\b(mountain|hill|upland)\b', 'upland')
    ]
    
    behavior_patterns = [
        (r'\b(ground|earth|floor)\b.*\b(feed|forage|search)\b', 'ground_feeder'),
        (r'\b(fly|flight|soar|glide)\b', 'flight'),
        (r'\b(sing|call|vocal)\b', 'vocal'),
        (r'\b(migrate|migratory)\b', 'migratory'),
        (r'\b(nest|breed)\b', 'breeding')
    ]
    
    text_lower = text.lower()
    
    all_patterns = [
        (size_patterns, 'size'),
        (color_patterns, 'color'),
        (body_patterns, 'anatomy'),
        (habitat_patterns, 'habitat'),
        (behavior_patterns, 'behavior')
    ]
    
    for pattern_group, category in all_patterns:
        for pattern, trait in pattern_group:
            matches = re.findall(pattern, text_lower)
            if matches:
                characteristics.extend([f"{category}:{trait}" for _ in matches])
    
    return list(set(characteristics))

def create_searchable_document(bird_data):
    """Create an enhanced document optimized for descriptive search."""
    title = bird_data.get('title', '')
    extract = bird_data.get('extract', '')
    description = bird_data.get('description', '')
    original_search = bird_data.get('original_search', title)

    primary_name = original_search if original_search else title
    full_text = f"{primary_name}\n{description}\n{extract}"
    characteristics = extract_characteristics(full_text)
    char_summary = " ".join(characteristics)
    
    search_traits = [primary_name.lower()]
    search_traits.extend([c.split(':')[1] for c in characteristics if c.startswith('color:')])
    search_traits.extend([c.split(':')[1] for c in characteristics if c.startswith('habitat:')])
    search_traits.extend([c.split(':')[1] for c in characteristics if c.startswith('anatomy:')])
    search_traits.extend([c.split(':')[1] for c in characteristics if c.startswith('behavior:')])

    enhanced_doc = f"""
SPECIES: {primary_name}

PHYSICAL_DESCRIPTION: {description}

CHARACTERISTICS: {char_summary}

DETAILED_INFO: {extract}

SEARCHABLE_TRAITS: {' '.join(search_traits)}
""".strip()
    
    return enhanced_doc, characteristics

def process_wikipedia_data(bird_data, doc_id):
    """Enhanced Wikipedia bird data processing."""
    if not bird_data.get('type') == 'wikipedia_bird_full':
        return None, None, None

    extract = bird_data.get('extract')
    if not extract or len(extract) < 100:
        return None, None, None

    enhanced_doc, characteristics = create_searchable_document(bird_data)
    primary_title = bird_data.get('original_search', bird_data.get('title', ''))

    metadata = {
        'title': str(primary_title),
        'family': "Unknown",
        'region': 'Europe',
        'source': 'Wikipedia',
        'url': str(bird_data.get('page_url', '')),
        'thumbnail': str(bird_data.get('thumbnail', '')),
        'search_name': str(bird_data.get('original_search', bird_data.get('title', ''))),
        'audio_url': str(bird_data.get('audio_url', '')),
        'type': 'bird_description',
        'characteristics': '|'.join(characteristics),
        'has_image': 'yes' if bird_data.get('thumbnail') else 'no',
        'has_audio': 'yes' if bird_data.get('audio_url') else 'no'
    }
    
    return enhanced_doc, metadata, f"bird_{doc_id:03d}"

def process_youtube_data(video_data, doc_id):
    """Process YouTube video data."""
    if not video_data.get('type') == 'youtube_video':
        return None, None, None

    doc_text = video_data.get('transcript')
    if not doc_text or len(doc_text) < 100:
        return None, None, None

    metadata = {
        'title': video_data.get('title', ''),
        'author': video_data.get('author', ''),
        'url': video_data.get('url', ''),
        'source': 'YouTube',
        'type': 'youtube_video',
        'video_id': video_data.get('video_id', '')
    }
    
    return doc_text, metadata, f"youtube_{doc_id:03d}"

def load_combined_data():
    """Load combined Wikipedia and YouTube data."""
    project_root = Path(__file__).resolve().parent.parent.parent
    data_file = project_root / "data" / "raw" / "combined_data.json"
    
    if not data_file.exists():
        logger.error(f"Combined data file not found at {data_file}. Run data collection first.")
        return []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    return all_data

def populate_chromadb():
    """Populate ChromaDB with enhanced bird data."""
    all_data = load_combined_data()
    if not all_data:
        logger.error("No data to process")
        return False
    
    logger.info(f"Found {len(all_data)} items in the data file. Starting processing...")
    chroma_client = ChromaClient()

    chroma_client.delete_collection("birds")
    chroma_client.delete_collection("youtube")
    
    bird_docs, bird_metadatas, bird_ids = [], [], []
    youtube_docs, youtube_metadatas, youtube_ids = [], [], []

    bird_counter, youtube_counter = 0, 0
    characteristics_stats = {}
    
    for item in all_data:
        if item.get('type') == 'wikipedia_bird_full':
            doc_text, metadata, doc_id = process_wikipedia_data(item, bird_counter)
            if doc_text:
                bird_docs.append(doc_text)
                bird_metadatas.append(metadata)
                bird_ids.append(doc_id)
                bird_counter += 1
                
                chars = metadata.get('characteristics', '').split('|')
                for char in chars:
                    if char:
                        characteristics_stats[char] = characteristics_stats.get(char, 0) + 1
                        
        elif item.get('type') == 'youtube_video':
            doc_text, metadata, doc_id = process_youtube_data(item, youtube_counter)
            if doc_text:
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
            logger.info(f"Added {len(bird_docs)} enhanced bird documents")
            
            top_characteristics = sorted(characteristics_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"Top characteristics found: {top_characteristics}")
        
        if youtube_docs:
            chroma_client.add_data(
                collection_name="youtube",
                documents=youtube_docs,
                metadata=youtube_metadatas,
                ids=youtube_ids
            )
            logger.info(f"Added {len(youtube_docs)} YouTube documents")
        
        return True
    except Exception as e:
        logger.error(f"Failed to populate database: {e}")
        return False

if __name__ == "__main__":
    success = populate_chromadb()
    if success:
        logger.info("Enhanced database population complete.")
    else:
        logger.error("Database population failed. Check errors above.")