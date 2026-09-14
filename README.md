# 🎯 Project Gandalf — Striver A2Z DSA Lecture Navigator

[![Release v1.1](https://img.shields.io/badge/Release-v1.1-brightgreen.svg)](https://github.com/krshnsaboo/Project-Gandalf/releases/tag/v1.1)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-gandalf.streamlit.app)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/krshnsaboo/strivera2z)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Live Demo:** https://project-gandalf.streamlit.app

Project Gandalf is an AI-powered **video retrieval and navigation system** for [Striver's A2Z DSA Course](https://takeuforward.org/strivers-a2z-dsa-course/).

Instead of generating solutions, Gandalf helps you **find where a concept or problem is actually explained**. Enter a DSA topic, concept, or LeetCode problem and get the most relevant lecture with an approximate timestamp and playable video.

Gandalf is designed as a **guide, not an answer machine**.

Rather than replacing the learning process with generated solutions, it helps students quickly navigate hundreds of hours of DSA lectures and reach the relevant explanation.

---

## ⚡ Features

- 🔍 **Semantic + Keyword Search** for accurate lecture retrieval
- 🧠 **Query Understanding** for DSA terminology, abbreviations, and problem names
- 🔗 **LeetCode URL Support** — paste a problem URL directly
- 🎯 **Relevant Timestamp Navigation** to jump to the useful part of a lecture
- 📺 **In-App Video Playback** through a Streamlit interface
- 💻 **Web & CLI Interfaces**
- 🛡️ **Robust Retrieval Pipeline** with reranking and fallback handling
- 📊 **Benchmarking & Evaluation** for measuring retrieval quality

---

## 🏗️ How It Works

```text
User Query
    │
    ▼
Query Understanding
    │
    ▼
Hybrid Retrieval
(Dense + Keyword Search)
    │
    ▼
Relevance Reranking
    │
    ▼
Relevant Lecture Segments
    │
    ▼
Navigation Response
    │
    ▼
Lecture + Timestamp + Video
````

The system combines multiple retrieval techniques to identify the most relevant parts of the course before generating a concise navigation response.

---

## 📊 Evaluation

Gandalf was evaluated on a curated set of **150 DSA queries** covering topics such as Arrays, Trees, Graphs, Dynamic Programming, Greedy, and Linked Lists.

| Metric                     |           Result |
| -------------------------- | ---------------: |
| **Recall@1**               |       **90.67%** |
| **Recall@5**               |       **98.67%** |
| **Recall@10**              |       **99.33%** |
| **MRR**                    |       **0.9421** |
| **Average Retrieval Time** | **~0.10s/query** |

These results measure how effectively the retrieval system identifies the lecture containing the expected explanation.

---

## 🚀 Quick Start

### Requirements

* Python 3.10+
* OpenAI API key for full navigation responses

### Installation

```bash
git clone https://github.com/krshnsaboo/Project-Gandalf.git
cd Project-Gandalf

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your-api-key
```

### Run

```bash
streamlit run app.py
```

Or use the CLI:

```bash
python main.py
```

---

## 🧪 Evaluation & Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the retrieval evaluation:

```bash
python evaluate.py
```

For reranked evaluation:

```bash
python evaluate.py --rerank
```

---

## 📁 Repository Overview

```text
Project Gandalf/
├── app.py                         # Streamlit application
├── main.py                        # CLI interface
├── rag_pipeline.py                # Main retrieval pipeline
├── retrieval.py                   # Search and retrieval
├── reranker.py                    # Result reranking
├── query_normalizer.py            # Query processing
├── prompt_builder.py              # Navigation response generation
├── response_parser.py             # Response processing
├── llm.py                         # LLM integration
├── evaluate.py                    # Evaluation
├── tests/                         # Unit tests
├── lecture_embeddings/            # Search index and metadata
└── scripts/                       # Data preprocessing utilities
```

---

## 📄 License & Acknowledgements

Released under the **MIT License**.

Course content is based on the [Striver A2Z DSA Course](https://takeuforward.org/strivers-a2z-dsa-course/) created by **Raj Vikramaditya (Striver)**.

This project is intended as a navigation and learning aid for the course content.
