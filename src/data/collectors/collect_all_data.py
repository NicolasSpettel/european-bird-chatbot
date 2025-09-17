import json
import logging
import time
from pathlib import Path

from src.data.collectors.bird_collector import ComprehensiveBirdCollector
from src.data.collectors.youtube_collector import YouTubeTranscriptCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent.parent.parent

def get_european_bird_list(file_path: Path) -> list:
    """Reads the list of European birds from the text file."""
    bird_list = []
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    names_on_line = line.split(',')
                    for name in names_on_line:
                        name = name.strip()
                        if name.startswith('"') and name.endswith('"'):
                            bird_name = name.strip('"')
                            bird_list.append(bird_name)
                        elif name:
                            logger.warning(f"Skipping improperly formatted name: {name}")
    except FileNotFoundError:
        logger.error(f"Error: The bird list file {file_path} was not found.")
        return []
    return bird_list

def collect_and_save_bird_data():
    """Collects comprehensive bird data for a predefined list of European birds and saves it to a JSON file."""
    logger.info("Starting bird data collection.")

    bird_list_path = PROJECT_ROOT / 'docs' / 'list_of_birds.txt'
    output_path = PROJECT_ROOT / 'data' / 'raw' / 'combined_data.json'

    bird_collector = ComprehensiveBirdCollector()
    youtube_collector = YouTubeTranscriptCollector()

    bird_names = get_european_bird_list(bird_list_path)
    if not bird_names:
        logger.error("No bird names found in the list. Exiting.")
        return

    all_bird_data = []

    logger.info("Collecting Wikipedia and audio data...")
    for name in bird_names:
        logger.info(f"Collecting data for {name}...")
        try:
            bird_data = bird_collector.collect_bird_data(name)
            if bird_data:
                bird_data['type'] = 'wikipedia_bird_full'
                all_bird_data.append(bird_data)
        except Exception as e:
            logger.error(f"Failed to collect data for {name}: {e}")

        time.sleep(1)

    logger.info(f"Finished collecting data for {len(all_bird_data)} birds from Wikipedia.")
    
    logger.info("Collecting YouTube video data...")
    
    youtube_urls = [
        "https://www.youtube.com/watch?v=DSIhFy6tlvI",
        "https://www.youtube.com/watch?v=NW9MVJmoRqQ",
        "https://www.youtube.com/watch?v=1RK4Nx4i924",
        "https://www.youtube.com/watch?v=hN7926wHsLk",
        "https://www.youtube.com/watch?v=NQoYVNAmpqE",
        "https://www.youtube.com/watch?v=WyB0QuFGiYU",
        "https://www.youtube.com/watch?v=22CzenMh5_k",
        "https://www.youtube.com/watch?v=TtATNKwUzg0"
    ]
    
    for url in youtube_urls:
        logger.info(f"Collecting YouTube data for URL: {url}...")
        try:
            video_data = youtube_collector.collect_video_data(url)
            if video_data:
                all_bird_data.append(video_data)
        except Exception as e:
            logger.error(f"Failed to collect YouTube data for {url}: {e}")
            
    logger.info(f"Finished collecting YouTube video data.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_bird_data, f, indent=4)
        logger.info(f"Successfully saved combined data to {output_path}")
    except IOError as e:
        logger.error(f"Failed to save data to file: {e}")

if __name__ == "__main__":
    collect_and_save_bird_data()