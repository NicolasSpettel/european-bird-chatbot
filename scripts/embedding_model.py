import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from typing import List

def create_examples_from_json(file_path: str) -> List[InputExample]:
    """
    Reads the JSON data and creates a list of InputExample objects.
    
    The function handles two types of data:
    1. Wikipedia entries: It pairs the 'title' with the 'extract' and 'description'.
    2. YouTube video transcripts: It pairs the video 'title' with the 'content' transcript.
    
    This creates positive pairs where the sentences are semantically related.
    """
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item.get("type") == "wikipedia_bird":
            # For Wikipedia data, pair the title with the extract and description.
            title = item.get("title")
            extract = item.get("extract")
            description = item.get("description")

            if title and extract:
                examples.append(InputExample(texts=[title, extract]))
            if title and description:
                examples.append(InputExample(texts=[title, description]))

        elif item.get("type") == "youtube_chunk":
            # For YouTube data, pair the title with the content (transcript).
            metadata = item.get("metadata", {})
            title = metadata.get("title")
            content = item.get("content")

            if title and content:
                examples.append(InputExample(texts=[title, content]))
    
    return examples

if __name__ == "__main__":
    json_file = 'data/raw/combined_data.json'

    train_examples = create_examples_from_json(json_file)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    model_name = 'all-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    num_epochs = 1
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of training steps

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path='./fine_tuned_model',
        show_progress_bar=True
    )