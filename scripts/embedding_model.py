# File: src/training/train_model.py

import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from typing import List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_examples_from_json(file_path: str) -> List[InputExample]:
    """
    Reads the JSON data and creates a list of InputExample objects.
    
    The function handles two types of data:
    1. Wikipedia entries: It pairs the 'title' with the 'extract' and 'description'.
    2. YouTube video transcripts: It pairs the video 'title' with the 'transcript'.
    
    This creates positive pairs where the sentences are semantically related.
    """
    examples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
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

        # This section is updated to match your data
        elif item.get("type") == "youtube_video":
            title = item.get("title")
            transcript = item.get("transcript")

            if title and transcript and len(transcript.split()) > 10:
                examples.append(InputExample(texts=[title, transcript]))
    
    logging.info(f"Created {len(examples)} training examples.")
    return examples

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    json_file = project_root / 'data' / 'raw' / 'combined_data.json'

    train_examples = create_examples_from_json(json_file)
    if not train_examples:
        logging.error("No training data available. Exiting.")
        exit()

    batch_size = 32
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    model_name = 'all-MiniLM-L6-v2'
    logging.info(f"Loading pre-trained model: {model_name}")
    model = SentenceTransformer(model_name)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    num_epochs = 2 
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    logging.info(f"Starting fine-tuning for {num_epochs} epochs with a batch size of {batch_size}.")
    logging.info(f"Warmup steps: {warmup_steps}")

    output_path = project_root / 'fine_tuned_model'
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=True,
    )
    
    logging.info(f"Fine-tuning complete. Model saved to '{output_path}'.")