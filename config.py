import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in .env")

# Change this whenever you want to switch models
OPENAI_MODEL = "gpt-4.1-mini"

# Retrieval settings
FAISS_TOP_K = 30
RERANK_TOP_K = 5

# Generation settings
MAX_TOKENS = 700
TEMPERATURE = 0.2