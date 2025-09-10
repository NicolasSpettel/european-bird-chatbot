import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from typing import Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)


class YouTubeTranscriptCollector:
    def __init__(self):
        self.headers = {
            "User-Agent": "BirdwatchingChatbot/1.0 Educational Project"
        }
        # Create a reusable API client instance
        self.ytt_api = YouTubeTranscriptApi()

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        logger.error(f"Could not extract video ID from: {url}")
        return None

    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """Get basic video info using oEmbed API"""
        try:
            oembed_url = (
                f"https://www.youtube.com/oembed?url="
                f"https://www.youtube.com/watch?v={video_id}&format=json"
            )
            response = requests.get(oembed_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get video info for {video_id}: {e}")
            return None

    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript for a YouTube video"""
        try:
            # Fetch transcript in English (manual or auto-generated)
            fetched_transcript = self.ytt_api.fetch(
                video_id, languages=["en", "en-US"]
            )
            raw_transcript = fetched_transcript.to_raw_data()

            # Join transcript lines into one string
            transcript = " ".join(entry["text"] for entry in raw_transcript)
            return transcript.strip()

        except NoTranscriptFound:
            logger.warning(
                f"No English transcript or captions found for video ID: {video_id}"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to get transcript for {video_id}: {e}")
            return None

    def collect_video_data(self, url: str, category: str = "birdwatching") -> Optional[Dict]:
        """Collect complete video data including transcript"""
        video_id = self.extract_video_id(url)
        if not video_id:
            return None

        video_info = self.get_video_info(video_id)
        transcript = self.get_transcript(video_id)

        if not transcript:
            logger.warning(f"No transcript available for: {url}. Skipping.")
            return None

        return {
            "video_id": video_id,
            "url": url,
            "title": video_info.get("title", "") if video_info else "",
            "author": video_info.get("author_name", "") if video_info else "",
            "transcript": transcript,
            "category": category,
            "type": "youtube_video",
        }
