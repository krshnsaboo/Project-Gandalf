import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except ImportError:
    st = None


def get_openai_api_key():
    # Local development (.env)
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return api_key

    # Streamlit Cloud (Secrets)
    if st is not None:
        try:
            return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

    raise ValueError(
        "OPENAI_API_KEY not found. Configure it in a .env file "
        "or in Streamlit Secrets."
    )


OPENAI_API_KEY = get_openai_api_key()

# OpenAI model
OPENAI_MODEL = "gpt-4.1-mini"

# Retrieval settings
FAISS_TOP_K = 30
RERANK_TOP_K = 5

# Generation settings
MAX_TOKENS = 700
TEMPERATURE = 0.2
