import os
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from huggingface_hub import create_repo

# Set up the Hugging Face authentication token
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

# Function to push models and tokenizers to Hugging Face Hub
def push_model_to_hub(repo_name, model_path, tokenizer_path):
    try:
        repo_url = create_repo(repo_name, private=True, token=HF_AUTH_TOKEN)
        print(f'{repo_url} has been successfully created.')
        model = AutoModel.from_pretrained(model_path)
        model.push_to_hub(repo_name, private=True, token=HF_AUTH_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.push_to_hub(repo_name, private=True, token=HF_AUTH_TOKEN)
    except Exception as e:
        print(f'Error during pushing {model_path}. Repo exists or an error occurred, check error: {e}')