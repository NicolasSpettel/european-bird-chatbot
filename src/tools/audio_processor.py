import os
import tempfile
from pathlib import Path
from typing import Optional
import logging

from openai import OpenAI
from src.config import Config

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handle audio input using OpenAI Whisper"""
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.webm', '.mp4']
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file to text using Whisper"""
        try:
            # Check file exists and format
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return None
            
            file_ext = Path(audio_file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported audio format: {file_ext}")
                return None
            
            # Transcribe using Whisper
            with open(audio_file_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"  # Optimize for English
                )
            
            transcribed_text = transcript.text.strip()
            logger.info(f"Audio transcribed successfully: '{transcribed_text[:50]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return None
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Optional[str]:
        """Transcribe audio from bytes (for web uploads)"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Transcribe
            result = self.transcribe_audio(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Audio bytes transcription failed: {e}")
            return None