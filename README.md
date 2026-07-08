# Project Gandalf

An AI-powered semantic navigation system for Striver's A2Z DSA Course.

Instead of answering DSA questions directly, Project Gandalf retrieves the most relevant lecture segment and guides users to the exact timestamp in the original YouTube lecture.

---

## Live Demo

🌐 https://project-gandalf.streamlit.app

---

## Features

- Semantic search across 315 Striver A2Z lecture transcripts
- Dense retrieval using FAISS and BAAI/bge-m3 embeddings
- Cross-encoder reranking using BAAI/bge-reranker-v2-m3
- GPT-5.5 powered natural-language navigation
- Direct YouTube timestamp links for one-click navigation
- Streamlit-based web interface
- Retrieval evaluation using Recall@K and MRR

---

## Architecture

```
User Query
      │
      ▼
BAAI/bge-m3 Embeddings
      │
      ▼
FAISS Retrieval
      │
      ▼
Cross Encoder Reranker
      │
      ▼
GPT-5.5
      │
      ▼
Lecture Recommendation
      │
      ▼
YouTube Timestamp Navigation
```

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python |
| Embedding Model | BAAI/bge-m3 |
| Vector Search | FAISS |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | GPT-5.5 (OpenAI API) |
| Frontend | Streamlit |

---

## Evaluation

Evaluation was performed on a manually curated benchmark consisting of 200 representative DSA queries.

| Metric | Score |
|--------|-------|
| Recall@1 | 90% |
| Recall@5 | 100% |
| Recall@10 | 100% |
| MRR | 0.95 |

---
## Dataset

- 315 lecture transcripts
- 5,752 indexed transcript chunks
- Dense vector retrieval with FAISS
- Direct timestamp navigation to the original YouTube lectures
- The processed dataset is also available on Kaggle for reproducibility and experimentation.

**Kaggle Dataset:** https://www.kaggle.com/datasets/krshnsaboo/strivera2z
 
---

## Performance

Measured on the development machine using CPU inference.

- Embedding + Retrieval: ~0.2 s
- Cross-Encoder Reranking: ~5 s
- GPT-5.5 Response Generation: ~2 s
- End-to-End Response Time: ~6–7 s
---
  
## Repository Structure

```
Project-Gandalf
│
├── app.py
├── rag_pipeline.py
├── retrieval.py
├── reranker.py
├── llm.py
├── prompt_builder.py
├── create_chunks.py
├── create_faiss_index.py
├── evaluate.py
├── lecture_embeddings/
├── final_jsons/
└── ...
```

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd Project-Gandalf
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create an environment file.

```text
OPENAI_API_KEY=your_api_key_here
```

Run the application.

```bash
streamlit run app.py
```

---

## Example

**Query**

```
I want to revise Kadane's Algorithm
```

**Output**

```
Lecture:
Kadane's Algorithm | Maximum Subarray Sum

Timestamp:
12:24

Watch from this timestamp
```

Selecting the recommendation opens the original YouTube lecture at the suggested timestamp.

---

## Motivation

This project was built to explore practical Retrieval-Augmented Generation (RAG) systems beyond question answering. The focus was on semantic retrieval, reranking, and improving navigation across long-form educational video content while preserving the original teaching material.

## Why "Project Gandalf"?
Like Gandalf in The Lord of the Rings, the system acts as a guide rather than providing all the answers itself. Instead of solving DSA problems, it directs learners to the exact point in Striver's lectures where the concept is explained.

---

## Future Improvements

- Larger evaluation benchmark
- Faster reranking
- Query expansion
- Deployment for public access
