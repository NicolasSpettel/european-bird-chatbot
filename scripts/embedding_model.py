# File: src/training/train_model.py

import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from typing import List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_examples_from_json(file_path: Path) -> List[InputExample]:
    """Creates InputExample objects from JSON data for training semantic similarity."""
    examples = []
    
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return []

    for item in data:
        if item.get("type") == "wikipedia_bird_full":
            title = item.get("title")
            extract = item.get("extract")
            description = item.get("description")

            if title and extract and len(extract.split()) > 10:
                examples.append(InputExample(texts=[title, extract]))
            if title and description and len(description.split()) > 10:
                examples.append(InputExample(texts=[title, description]))

        elif item.get("type") == "youtube_video":
            title = item.get("title")
            transcript = item.get("transcript")

            if title and transcript and len(transcript.split()) > 10:
                examples.append(InputExample(texts=[title, transcript]))
    
    logger.info(f"Created {len(examples)} training examples.")
    return examples

def train_bird_model():
    """Fine-tunes a sentence transformer model on bird-related data."""
    project_root = Path(__file__).resolve().parent.parent
    json_file = project_root / 'data' / 'raw' / 'combined_data.json'

    train_examples = create_examples_from_json(json_file)
    if not train_examples:
        logger.error("No training data available. Exiting.")
        return

    batch_size = 32
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    model_name = 'all-MiniLM-L6-v2'
    logger.info(f"Loading pre-trained model: {model_name}")
    model = SentenceTransformer(model_name)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    num_epochs = 2 
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    logger.info(f"Starting fine-tuning for {num_epochs} epochs with batch size {batch_size}")
    logger.info(f"Warmup steps: {warmup_steps}")

    output_path = project_root / 'fine_tuned_model'
    output_path.mkdir(exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=True,
    )
    
    logger.info(f"Fine-tuning complete. Model saved to '{output_path}'")

if __name__ == "__main__":
    train_bird_model()