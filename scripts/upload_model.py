# File: src/upload_model.py

import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_model_to_huggingface(local_model_path: Path, repo_name: str):
    """
    Loads the fine-tuned model from a local path and uploads it to Hugging Face.
    """
    try:
        logging.info(f"Loading model from {local_model_path}...")
        model = SentenceTransformer(str(local_model_path))
        
        logging.info(f"Uploading model to Hugging Face Hub as {repo_name}...")
        model.push_to_hub(repo_name)
        
        logging.info("Model upload complete!")

    except Exception as e:
        logging.error(f"Failed to upload model: {e}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    local_model_path = project_root / 'fine_tuned_model'

    huggingface_repo_name = "Nicolas-Spettel/bird-qa-model"

    upload_model_to_huggingface(local_model_path, huggingface_repo_name)