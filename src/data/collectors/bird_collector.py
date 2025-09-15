# File: src/data/collectors/bird_collector.py

import requests
import time
import logging
from typing import Dict, Optional, List
from src.config import Config

logger = logging.getLogger(__name__)

class XenoCantoCollector:
    """Collects bird audio data by making requests to the real Xeno-Canto API."""
    def get_best_recording_url(self, bird_name: str) -> Optional[str]:
        """
        Searches the Xeno-Canto API for a bird's audio recording.
        Prefers recordings with a high quality rating ('A').
        Includes a fallback search if the initial search fails.
        """
        api_url = "https://xeno-canto.org/api/2/recordings"
        
        search_queries = [bird_name.lower()]
        
        if len(bird_name.split()) > 1:
            search_queries.append(bird_name.split()[-1].lower())

        for query in search_queries:
            try:
                logger.info(f"Searching Xeno-Canto for audio with query: {query}")
                response = requests.get(
                    api_url,
                    params={'query': query},
                    headers={'User-Agent': 'EuropeanBirdChatbot/1.0'}
                )
                response.raise_for_status()
                data = response.json()

                if not data.get('recordings'):
                    logger.warning(f"No recordings found for query '{query}'. Trying next query...")
                    continue

                recordings = data['recordings']
                best_recording = next((rec for rec in recordings if rec.get('q') == 'A'), None)
                if not best_recording and recordings:
                    best_recording = recordings[0]
                
                if best_recording:
                    audio_url = best_recording.get('file')
                    if audio_url:
                        logger.info(f"Found audio URL for {bird_name}: {audio_url}")
                        return audio_url

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to Xeno-Canto API for query '{query}': {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing Xeno-Canto response for query '{query}': {e}")
                continue
        
        logger.warning(f"Could not extract a valid audio URL for {bird_name} after all attempts.")
        return None

class ComprehensiveBirdCollector:
    """
    Orchestrates data collection from multiple sources for a single bird species.
    """
    def __init__(self):
        self.wiki_base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.headers = {
            'User-Agent': 'EuropeanBirdChatbot/1.0 Educational Project',
            'Accept': 'application/json'
        }
        self.xeno_canto_collector = XenoCantoCollector()
        # GBIF collector has been removed as it's no longer needed.
    
    def get_wikipedia_info(self, bird_name: str) -> Optional[Dict]:
        """Get bird info directly from Wikipedia page summary."""
        try:
            clean_name = bird_name.replace(' ', '_')
            url = f"{self.wiki_base_url}{clean_name}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 404:
                alt_name = bird_name.replace('Common ', '').replace('Eurasian ', '').replace('European ', '')
                clean_alt = alt_name.replace(' ', '_')
                alt_url = f"{self.wiki_base_url}{clean_alt}"
                response = requests.get(alt_url, headers=self.headers)
            
            response.raise_for_status()
            data = response.json()
            bird_info = {
                'title': data.get('title', bird_name),
                'description': data.get('description', ''),
                'extract': data.get('extract', ''),
                'thumbnail': data.get('thumbnail', {}).get('source', ''),
                'page_url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                'original_search': bird_name
            }
            if len(bird_info['extract']) > 50:
                logger.info(f"Successfully retrieved: {bird_info['title']}")
                return bird_info
            else:
                logger.warning(f"Insufficient content for: {bird_name}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {bird_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {bird_name}: {e}")
            return None

    def collect_bird_data(self, bird_name: str) -> Optional[Dict]:
        """
        Collects comprehensive data for a bird from all available sources.
        Combines Wikipedia and Xeno-Canto data into a single dictionary.
        """
        wiki_info = self.get_wikipedia_info(bird_name)
        
        if not wiki_info:
            return None

        # Add data from other collectors to the Wikipedia dictionary
        audio_url = self.xeno_canto_collector.get_best_recording_url(bird_name)
        wiki_info['audio_url'] = audio_url if audio_url else None
        
        # The GBIF data collection has been removed.
        
        # It's good practice to add a small delay to respect API rate limits
        time.sleep(0.5)
        return wiki_info