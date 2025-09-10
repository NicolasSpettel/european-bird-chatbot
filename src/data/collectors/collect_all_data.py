import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Ensure imports work from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.data.collectors.simple_collector import SimpleWikipediaCollector
from src.data.collectors.youtube_collector import YouTubeTranscriptCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# European birds for the chatbot
EUROPEAN_BIRDS = [
    "European Robin", "House Sparrow", "Common Blackbird", "Eurasian Blue Tit",
    "Great Tit", "Eurasian Wren", "Common Chaffinch", "European Goldfinch",
    "European Greenfinch", "Eurasian Bullfinch", "Song Thrush", "Common Redstart",
    "Eurasian Blackcap", "Coal Tit", "Eurasian Nuthatch", "Eurasian Treecreeper",
    "Common Starling", "Eurasian Magpie", "Eurasian Jay", "Carrion Crow",
    "Common Buzzard", "Red Kite", "Barn Owl", "Tawny Owl", "Common Kingfisher",
    "Great Spotted Woodpecker", "Green Woodpecker", "Common Swift", "Barn Swallow",
    "House Martin", "Common Linnet", "Yellowhammer", "Reed Bunting",
    "Common Whitethroat", "Garden Warbler", "Willow Warbler", "Common Chiffchaff",
    "Goldcrest", "Long-tailed Tit", "Marsh Tit", "Common Pheasant",
    "Grey Partridge", "Common Moorhen", "Eurasian Coot", "Northern Lapwing",
    "Common Snipe", "Common Redshank", "Herring Gull", "Common Tern", "Stock Dove"
]

YOUTUBE_VIDEOS = [
    {
        "url": "https://www.youtube.com/watch?v=1RK4Nx4i924",
        "category": "beginners_guide",
    },
        {
        "url": "https://www.youtube.com/watch?v=hK30ObyJt6M",
        "category": "beginners_guide",
    },
            {
        "url": "https://www.youtube.com/watch?v=uuY_7i040ug",
        "category": "beginners_guide",
    },
            {
        "url": "https://www.youtube.com/watch?v=js4Ir0ExlYQ",
        "category": "photography_guide",
    },

]

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks for embedding.
    - chunk_size: max characters per chunk
    - overlap: repeated characters between chunks for context
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap  # step forward with overlap

        if start < 0:
            start = 0

    return chunks


def process_youtube_data(video_data, doc_id):
    """Process YouTube video data into chunks"""
    transcript = video_data.get('transcript', '')
    if not transcript or len(transcript) < 100:
        return [], [], []

    # Split transcript into smaller chunks
    chunks = chunk_text(transcript, chunk_size=800, overlap=100)

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        doc_text = f"{video_data['title']}\n\n"
        if video_data.get('author'):
            doc_text += f"Author: {video_data['author']}\n\n"
        doc_text += f"Transcript (part {i+1}): {chunk}"

        metadata = {
            'title': video_data['title'],
            'author': video_data.get('author', ''),
            'category': video_data.get('category', 'birdwatching'),
            'source': 'YouTube',
            'url': video_data['url'],
            'video_id': video_data['video_id'],
            'chunk_index': i,
            'type': 'youtube_content'
        }

        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(f"youtube_{doc_id:03d}_chunk_{i:03d}")

    return documents, metadatas, ids


def collect_all_data():
    """Collect both Wikipedia and YouTube data and process them."""
    wiki_collector = SimpleWikipediaCollector()
    youtube_collector = YouTubeTranscriptCollector()

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    doc_id_counter = 0

    # Collect Wikipedia data
    for bird_name in EUROPEAN_BIRDS:
        try:
            bird_data = wiki_collector.collect_bird_data(bird_name)
            if bird_data:
                bird_data["type"] = "wikipedia_bird"
                all_data.append(bird_data)
        except Exception as e:
            logger.error(f"Failed to collect Wikipedia data for {bird_name}: {e}")

    # Collect and process YouTube data
    for video in YOUTUBE_VIDEOS:
        try:
            video_data = youtube_collector.collect_video_data(
                video["url"], video.get("category", "birdwatching")
            )
            if video_data:
                documents, metadatas, ids = process_youtube_data(video_data, doc_id_counter)
                doc_id_counter += 1

                for doc_text, metadata, doc_id in zip(documents, metadatas, ids):
                    all_data.append({
                        "id": doc_id,
                        "content": doc_text,
                        "metadata": metadata,
                        "type": "youtube_chunk"
                    })
        except Exception as e:
            logger.error(f"Failed to collect YouTube data for {video['url']}: {e}")

    # Save combined data
    output_file = data_dir / "combined_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(all_data)} records to {output_file}")
    return all_data


if __name__ == "__main__":
    collected_data = collect_all_data()
