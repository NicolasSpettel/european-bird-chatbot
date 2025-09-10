import requests
import json
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleWikipediaCollector:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.headers = {
            'User-Agent': 'EuropeanBirdChatbot/1.0 Educational Project',
            'Accept': 'application/json'
        }
    
    def get_bird_info(self, bird_name: str) -> Optional[Dict]:
        """Get bird info directly from Wikipedia page"""
        try:
            # Clean the bird name for URL
            clean_name = bird_name.replace(' ', '_')
            url = f"{self.base_url}{clean_name}"
            
            response = requests.get(url, headers=self.headers)
            
            # If direct name fails, try some variations
            if response.status_code == 404:
                # Try without "Common" or "Eurasian"
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
        """Collect bird data with rate limiting"""
        result = self.get_bird_info(bird_name)
        time.sleep(0.5)  # Rate limiting
        return result