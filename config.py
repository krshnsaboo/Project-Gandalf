import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except ImportError:
    st = None


def get_openai_api_key(required: bool = False) -> str | None:
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

OPENAI_MODEL = "gpt-4o-mini"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

FAISS_TOP_K = 30
RERANK_TOP_K = 5
RERANK_MIN_SCORE = -5.0

USE_HYBRID_SEARCH = True
RRF_K = 60

MAX_TOKENS = 700
TEMPERATURE = 0.2