import requests
import time
import logging
from typing import Dict, Optional

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
        
        # Create a list of potential search queries from the bird name
        search_queries = [bird_name.lower()]
        
        # Add a fallback query using the last word of the bird name
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
                
                # Prioritize recordings with the highest quality rating ('A')
                best_recording = next((rec for rec in recordings if rec.get('q') == 'A'), None)

                # If no 'A' quality recording is found, just use the first one
                if not best_recording and recordings:
                    best_recording = recordings[0]
                
                if best_recording:
                    # The 'file' field is a full URL, so we don't need to add the base URL again.
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

class SimpleWikipediaCollector:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.headers = {
            'User-Agent': 'EuropeanBirdChatbot/1.0 Educational Project',
            'Accept': 'application/json'
        }
        # Initialize the audio collector
        self.xeno_canto_collector = XenoCantoCollector()
    
    def get_bird_info(self, bird_name: str) -> Optional[Dict]:
        """Get bird info directly from Wikipedia page"""
        try:
            # Clean the bird name for URL
            clean_name = bird_name.replace(' ', '_')
            url = f"{self.base_url}{clean_name}"
            
            response = requests.get(url, headers=self.headers)
            
            # If direct name fails, try some variations
            if response.status_code == 404:
                alt_name = bird_name.replace('Common ', '').replace('Eurasian ', '').replace('European ', '')
                clean_alt = alt_name.replace(' ', '_')
                alt_url = f"{self.base_url}{clean_alt}"
                response = requests.get(alt_url, headers=self.headers)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract information
            bird_info = {
                'title': data.get('title', bird_name),
                'description': data.get('description', ''),
                'extract': data.get('extract', ''),
                'thumbnail': data.get('thumbnail', {}).get('source', ''),
                'page_url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                'original_search': bird_name
            }
            
            # Only return if we have substantial content
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
        """Collect bird data including an audio URL with rate limiting"""
        wiki_info = self.get_bird_info(bird_name)
        
        if wiki_info:
            # Now, get the audio URL using the same bird name
            audio_url = self.xeno_canto_collector.get_best_recording_url(bird_name)
            
            # Add the audio URL to the dictionary
            wiki_info['audio_url'] = audio_url if audio_url else None
            
        time.sleep(0.5)  # Rate limiting
        return wiki_info