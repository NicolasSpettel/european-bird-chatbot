import sys
import os
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Ensure imports work from src/
# Assuming bird_collector.py is the correct module name for SimpleWikipediaCollector
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.data.collectors.bird_collector import SimpleWikipediaCollector
from src.data.collectors.youtube_collector import YouTubeTranscriptCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# European birds for the chatbot
EUROPEAN_BIRDS = [
    # Waterfowl
    "Mute Swan", "Whooper Swan", "Bewick's Swan", "Greylag Goose", "Canada Goose", "Barnacle Goose",
    "Egyptian Goose", "Shelduck", "Mallard", "Gadwall", "Teal", "Pintail", "Shoveler", "Pochard",
    "Tufted Duck", "Scaup", "Eider", "Long-tailed Duck", "Goldeneye", "Smew", "Red-breasted Merganser",
    "Goosander", "Velvet Scoter",

    # Pheasants, Partridges, and Quails
    "Common Pheasant", "Grey Partridge", "Red-legged Partridge", "Common Quail",

    # Grebes
    "Little Grebe", "Great Crested Grebe", "Red-necked Grebe", "Slavonian Grebe", "Black-necked Grebe",

    # Pigeons and Doves
    "Rock Dove", "Stock Dove", "Wood Pigeon", "Collared Dove", "Turtle Dove",

    # Cuckoos
    "Common Cuckoo",

    # Nightjars
    "Nightjar",

    # Swifts
    "Common Swift", "Pallid Swift", "Alpine Swift",

    # Rails, Crakes, and Coots
    "Water Rail", "Spotted Crake", "Corn Crake", "Moorhen", "Coot",

    # Cranes
    "Common Crane",

    # Bustards
    "Great Bustard", "Little Bustard",

    # Oystercatchers
    "Eurasian Oystercatcher",

    # Plovers
    "Northern Lapwing", "Little Ringed Plover", "Ringed Plover", "Kentish Plover", "Golden Plover",
    "Grey Plover", "Dotterel",

    # Sandpipers and Snipes
    "Common Snipe", "Jack Snipe", "Woodcock", "Black-tailed Godwit", "Bar-tailed Godwit",
    "Whimbrel", "Curlew", "Spotted Redshank", "Redshank", "Greenshank", "Green Sandpiper",
    "Wood Sandpiper", "Common Sandpiper", "Turnstone", "Knot", "Sanderling", "Little Stint",
    "Temminck's Stint", "Dunlin", "Ruff", "Avocet",

    # Skuas and Gulls
    "Great Skua", "Arctic Skua", "Pomarine Skua", "Long-tailed Skua", "Mediterranean Gull",
    "Black-headed Gull", "Little Gull", "Common Gull", "Lesser Black-backed Gull",
    "Herring Gull", "Yellow-legged Gull", "Great Black-backed Gull", "Kittiwake", "Sabine's Gull",
    "Ivory Gull", "Glaucous Gull",

    # Terns
    "Little Tern", "Sandwich Tern", "Common Tern", "Arctic Tern", "Roseate Tern", "Black Tern",
    "White-winged Black Tern",

    # Auks
    "Guillemot", "Razorbill", "Black Guillemot", "Little Auk", "Puffin",

    # Storks
    "White Stork", "Black Stork",

    # Herons, Egrets, and Bitterns
    "Great Bittern", "Little Bittern", "Grey Heron", "Purple Heron", "Great Egret", "Little Egret",
    "Cattle Egret", "Squacco Heron", "Night Heron",

    # Ibises and Spoonbills
    "Glossy Ibis", "Spoonbill",

    # Birds of Prey
    "Honey Buzzard", "Black Kite", "Red Kite", "White-tailed Eagle", "Golden Eagle",
    "Short-toed Eagle", "Marsh Harrier", "Hen Harrier", "Montagu's Harrier", "Goshawk",
    "Sparrowhawk", "Buzzard", "Rough-legged Buzzard", "Osprey", "Kestrel", "Red-footed Falcon",
    "Merlin", "Hobby", "Peregrine Falcon",

    # Owls
    "Barn Owl", "Scops Owl", "Eagle Owl", "Tawny Owl", "Long-eared Owl", "Short-eared Owl",
    "Little Owl", "Pygmy Owl",

    # Kingfishers
    "Common Kingfisher",

    # Woodpeckers
    "Wryneck", "Green Woodpecker", "Great Spotted Woodpecker", "Middle Spotted Woodpecker",
    "Lesser Spotted Woodpecker", "Black Woodpecker", "Grey-headed Woodpecker",

    # Larks
    "Woodlark", "Skylark", "Crested Lark", "Shore Lark",

    # Swallows and Martins
    "Sand Martin", "Swallow", "House Martin", "Red-rumped Swallow",

    # Pipits and Wagtails
    "Tree Pipit", "Meadow Pipit", "Rock Pipit", "Water Pipit", "White Wagtail", "Grey Wagtail",
    "Yellow Wagtail",

    # Waxwings
    "Waxwing",

    # Dippers
    "White-throated Dipper",

    # Wrens
    "Eurasian Wren",

    # Accentors
    "Dunnock",

    # Robins and Chats
    "European Robin", "Common Nightingale", "Bluethroat", "Redstart", "Black Redstart",
    "Stonechat", "Whinchat", "Wheatear",

    # Thrushes
    "Ring Ouzel", "Blackbird", "Fieldfare", "Song Thrush", "Redwing", "Mistle Thrush",

    # Warblers
    "Cetti's Warbler", "Sedge Warbler", "Reed Warbler", "Marsh Warbler", "Icterine Warbler",
    "Willow Warbler", "Common Chiffchaff", "Wood Warbler", "Greenish Warbler", "Arctic Warbler",
    "Bonelli's Warbler", "Garden Warbler", "Blackcap", "Lesser Whitethroat", "Common Whitethroat",
    "Dartford Warbler", "Subalpine Warbler", "Sardinian Warbler", "Melodious Warbler",

    # Flycatchers
    "Spotted Flycatcher", "Pied Flycatcher", "Collared Flycatcher", "Red-breasted Flycatcher",

    # Tits
    "Coal Tit", "Marsh Tit", "Willow Tit", "Crested Tit", "Blue Tit", "Great Tit", "Long-tailed Tit",

    # Nuthatches
    "Eurasian Nuthatch",

    # Treecreepers
    "Eurasian Treecreeper", "Short-toed Treecreeper",

    # Shrikes
    "Red-backed Shrike", "Lesser Grey Shrike", "Woodchat Shrike",

    # Crows and Allies
    "Eurasian Jay", "Magpie", "Spotted Nutcracker", "Chough", "Jackdaw", "Rook", "Carrion Crow",
    "Hooded Crow", "Raven",

    # Starlings
    "European Starling", "Rose-coloured Starling",

    # Sparrows
    "House Sparrow", "Tree Sparrow",

    # Finches
    "Chaffinch", "Brambling", "Hawfinch", "Greenfinch", "Goldfinch", "Siskin", "Linnet",
    "Twite", "Common Redpoll", "Arctic Redpoll", "Two-barred Crossbill", "Common Crossbill",
    "Parrot Crossbill", "Bullfinch",

    # Buntings
    "Yellowhammer", "Cirl Bunting", "Rock Bunting", "Ortolan Bunting", "Reed Bunting",
    "Corn Bunting", "Lapland Bunting", "Snow Bunting",

    # Miscellaneous (removed duplicates)
    "Bearded Reedling", "Penduline Tit", "Golden Oriole", "Hoopoe",
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

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 100):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= text_length:
            break
        start = end - overlap
    return chunks



def process_youtube_data(video_data, doc_id):
    """Process YouTube video data into chunks"""
    transcript = video_data.get('transcript', '')
    logger.info(f"Transcript length: {len(transcript)}")
    if not transcript or len(transcript) < 100:
        return [], [], []

    # Split transcript into smaller chunks
    # no more than 1million characters
    transcript = transcript[:1000000]
    chunks = chunk_text(transcript, chunk_size=1400, overlap=100)

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        doc_text = f"{video_data.get('title','')}\n\n"
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

    for video in YOUTUBE_VIDEOS:
        try:
            logger.info(f"Processing YouTube video: {video['url']}")
            video_data = youtube_collector.collect_video_data(
                video["url"], video.get("category", "birdwatching")
            )
            if not video_data:
                logger.warning(f"No data collected for: {video['url']}")
                continue
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
            logger.error(f"Failed to collect YouTube data for {video['url']}: {e}\n{traceback.format_exc()}")

    # Save combined data
    output_file = data_dir / "combined_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(all_data)} records to {output_file}")
    return all_data


if __name__ == "__main__":
    collected_data = collect_all_data()