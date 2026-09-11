import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except ImportError:
    st = None


def get_openai_api_key(required: bool = False) -> str | None:
    """
    Retrieve OPENAI_API_KEY from environment variables or Streamlit secrets.
    If required=True, raises ValueError when not found.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return api_key

    if st is not None:
        try:
            return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

    if required:
        raise ValueError(
            "OPENAI_API_KEY not found. Configure it in a .env file "
            "or in Streamlit Secrets."
        )

    return None


OPENAI_API_KEY = get_openai_api_key(required=False)

# Model Settings
OPENAI_MODEL = "gpt-4o-mini"

# Fast CPU-optimized cross-encoder (22M params, ~50-100ms vs 16s with bge-reranker-v2-m3)
# To use the heavy model on GPU, set: "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Settings
FAISS_TOP_K = 30
RERANK_TOP_K = 5
RERANK_MIN_SCORE = -5.0  # Threshold to prune irrelevant reranked candidates

# Hybrid Search Settings
USE_HYBRID_SEARCH = True
RRF_K = 60  # Reciprocal Rank Fusion constant

# Generation Settings
MAX_TOKENS = 700
TEMPERATURE = 0.2