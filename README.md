# 🎯 Project Gandalf: Striver A2Z DSA Lecture Navigator

[![Release v1.1](https://img.shields.io/badge/Release-v1.1-brightgreen.svg)](https://github.com/krshnsaboo/Project-Gandalf/releases/tag/v1.1)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-gandalf.streamlit.app)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/krshnsaboo/strivera2z)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Live Demo**: [https://project-gandalf.streamlit.app](https://project-gandalf.streamlit.app)  
> **Kaggle Dataset**: [https://www.kaggle.com/datasets/krshnsaboo/strivera2z](https://www.kaggle.com/datasets/krshnsaboo/strivera2z)

**Project Gandalf** is an AI-powered Video Retrieval-Augmented Generation (RAG) navigation engine designed specifically for [Striver's A2Z DSA Course](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2) (315 lectures on YouTube).

### 🧙 Why "Project Gandalf"?
Like Gandalf in *The Lord of the Rings*, the system acts as a **wise guide rather than an answer machine**. Instead of generating abstract code solutions or forcing students to scrub through hundreds of hours of video, Gandalf pinpoints the **exact lecture, timestamp, and playable in-app video player** where Raj Vikramaditya (Striver) explains the intuition, algorithm, or problem.

---

## ⚡ Key Capabilities

- **🔍 Hybrid Search Engine**: Integrates dense semantic vector search via **FAISS** (`BAAI/bge-m3`, 1024 dimensions) with sparse keyword retrieval via **BM25Okapi**, blended seamlessly using **Reciprocal Rank Fusion (RRF, $k=60$)**.
- **🚀 Sub-Second Cross-Encoder Reranking**: Re-scores top candidate passages using `cross-encoder/ms-marco-MiniLM-L-6-v2` with candidate pre-pruning ($k=20$) and relevance thresholding (`-5.0`), delivering precise rankings in ~50–100ms on standard CPUs.
- **🧠 Query Enhancement & Terminology Expansion**: Automatically detects and expands shorthand technical terms (`pq` $\to$ Priority Queue, `dp` $\to$ Dynamic Programming, `bst` $\to$ Binary Search Tree, `dfs/bfs` $\to$ Depth/Breadth First Search, `mst` $\to$ Minimum Spanning Tree, `ll` $\to$ Linked List, `tc/sc` $\to$ Time/Space Complexity) and structures intent for clearer retrieval.
- **🔗 LeetCode URL Auto-Normalization**: Directly extracts and normalizes problem titles from LeetCode URLs (e.g., `https://leetcode.com/problems/lru-cache/` $\to$ `LRU Cache`, `course-schedule-ii` $\to$ `Course Schedule II`) before querying.
- **⚡ Token-Optimized Navigation Prompting**: Streamlined, high-density prompt design that slashes prompt token consumption by **>60%** while strictly enforcing factual timestamp extraction without algorithm hallucinations.
- **📺 Interactive Streamlit Web UI**: Clean, distraction-free web interface featuring a centered search layout, instant timestamped YouTube video player, and one-click popular search chips.
- **🖥️ Dual Interface**: Supports both a modern Streamlit web application and an interactive terminal CLI (`main.py`).
- **📊 Telemetry & Observability**: Real-time structured query logging (`logs/search_logs.jsonl`) recording retrieval, reranking, and generation latency breakdowns alongside ranking metrics.
- **🛡️ Built-in Resilience**: Automatic exponential retry policies on OpenAI API calls and zero-downtime graceful fallback to the top reranked context during external API outages.

---

## 📊 Benchmark Results (150-Query Evaluation Suite)

Project Gandalf was quantitatively benchmarked against a ground-truth dataset of 150 diverse DSA questions across all topics (Arrays, Trees, Graphs, Dynamic Programming, Greedy, Linked Lists, etc.):

| Metric | Score | Performance Details |
| :--- | :---: | :--- |
| **Recall@1** | **90.67%** | **136 out of 150 queries** matched the exact lecture at rank #1 |
| **Recall@5** | **98.67%** | **148 out of 150 queries** found the target lecture in the top 5 |
| **Recall@10** | **99.33%** | **149 out of 150 queries** found the target lecture in the top 10 |
| **Mean Reciprocal Rank (MRR)** | **0.9421** | Extremely high rank precision across complex queries |
| **Retrieval Throughput** | **~0.10s / query** | Complete 150-query suite evaluates in ~18 seconds |

---

## 📁 Repository Structure

```text
Project Gandalf/
├── app.py                     # Streamlit web application with embedded video player
├── main.py                    # Interactive CLI search tool
├── rag_pipeline.py            # End-to-end RAG orchestrator with query enhancement
├── retrieval.py               # Hybrid Search: BM25 (sparse) + FAISS (dense) via RRF
├── reranker.py                # Cross-encoder reranker with candidate pre-pruning
├── prompt_builder.py          # Token-optimized navigation prompts & timestamp formatting
├── llm.py                     # OpenAI client with retries & graceful fallback
├── response_parser.py         # Structured parsing of recommendations and timestamps
├── query_normalizer.py        # Extracts LeetCode titles, expands acronyms & enhances queries
├── config.py                  # Centralized configuration & lazy secrets loader
├── logger.py                  # Telemetry logger writing structured JSONL records
├── analyse_logs.py            # Telemetry analysis script for latencies and query stats
├── evaluate.py                # Retrieval & pipeline benchmark evaluation runner
├── evaluation_queries.json    # 150 curated benchmark test cases
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
│
├── tests/                     # Automated unit test suite (20 unit tests)
│   ├── test_response_parser.py
│   ├── test_query_normalizer.py
│   ├── test_prompt_builder.py
│   └── test_config.py
│
├── lecture_embeddings/        # Runtime vector indices
│   ├── faiss_index.bin        # 5,752 indexed passage vectors
│   ├── all_lecture_embeddings.pkl # Complete chunks & metadata
│   └── faiss_info.json        # Index metadata
│
└── scripts/                   # Data preprocessing pipelines
    ├── videos_metadata.py     # Playlist metadata scraper using yt-dlp
    ├── create_chunks.py       # Whisper ASR transcription
    ├── merge_chunks.py        # Subtitle grouping & token bounding
    ├── absorb_tiny_chunks.py  # Tiny chunk absorption
    ├── chunk_analyzer.py      # Statistical inspection of chunks
    └── create_faiss_index.py  # FAISS index builder
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- An OpenAI API Key (optional for pure retrieval/fallback mode, recommended for full synthesis)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/krshnsaboo/Project-Gandalf.git
cd "Project Gandalf"

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```
*(Note: If running on Streamlit Cloud, you can also provide `OPENAI_API_KEY` via Streamlit Secrets.)*

---

## 💻 Running the Application

### Launch Streamlit Web UI
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Enter any DSA concept (e.g. *"Dijkstra with Priority Queue"*, *"Trapping Rain Water Optimal"*) or paste a LeetCode problem URL to view the exact video timestamp with playable in-app playback.

### Run Interactive CLI
```powershell
python main.py
```

### Run Automated Unit Tests
```powershell
python -m unittest discover -s tests -v
```

### Run Evaluation Benchmark
```powershell
# Evaluate retrieval stage (Dense + BM25 RRF)
python evaluate.py

# Evaluate full pipeline with Cross-Encoder Reranking
python evaluate.py --rerank
```

### Analyze Search Telemetry
```powershell
python analyse_logs.py
```

---

## ⚙️ Configuration Reference (`config.py`)

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | OpenAI model used for timestamp navigation responses |
| `RERANKER_MODEL` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model (switch to `"BAAI/bge-reranker-v2-m3"` for heavy GPU setups) |
| `FAISS_TOP_K` | `30` | Candidate pool size retrieved from dense vector search |
| `RERANK_TOP_K` | `5` | Top reranked segments presented to the prompt builder |
| `RERANK_MIN_SCORE` | `-5.0` | Logit threshold to prune irrelevant candidate chunks |
| `USE_HYBRID_SEARCH` | `True` | Enable combined BM25 + FAISS Reciprocal Rank Fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion smoothing constant |
| `MAX_TOKENS` | `700` | Maximum token limit for LLM generation |
| `TEMPERATURE` | `0.2` | Sampling temperature for factual timestamp extraction |

---

## 📄 License & Acknowledgements

- **Course Content**: Special thanks to **Raj Vikramaditya (Striver)** for creating the [TakeUForward A2Z DSA Course](https://takeuforward.org).
- **License**: Released under the [MIT License](LICENSE).
