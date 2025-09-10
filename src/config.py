# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    
    # Database
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # European Countries for filtering
    EUROPEAN_COUNTRIES = [
        "United Kingdom", "Ireland", "France", "Spain", "Portugal", "Italy",
        "Germany", "Netherlands", "Belgium", "Luxembourg", "Switzerland", 
        "Austria", "Denmark", "Sweden", "Norway", "Finland", "Poland",
        "Czech Republic", "Slovakia", "Hungary", "Romania", "Bulgaria",
        "Greece", "Croatia", "Slovenia", "Estonia", "Latvia", "Lithuania"
    ]