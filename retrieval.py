import pickle
import re
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import USE_HYBRID_SEARCH, RRF_K


class Retriever:

    def __init__(
        self,
        embeddings_path: str = "lecture_embeddings/all_lecture_embeddings.pkl",
        index_path: str = "lecture_embeddings/faiss_index.bin",
        model_name: str = "BAAI/bge-m3",
    ):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)

        print("Loading metadata...")
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)

        self.metadata = []
        for lecture in data:
            self.metadata.extend(lecture["metadata"])

        print(f"Loaded {len(self.metadata)} chunks.")
        print(f"Embedding dimension: {self.index.d}")
        print(f"Indexed vectors: {self.index.ntotal}")

        # Build in-memory BM25 index for Hybrid Search
        print("Building BM25 sparse index...")
        t_bm25_start = time.perf_counter()
        self.corpus_tokens = [
            self._tokenize(f"{item.get('lecture_title', '')} {item.get('text', '')}")
            for item in self.metadata
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        t_bm25_end = time.perf_counter()
        print(f"BM25 index built in {t_bm25_end - t_bm25_start:.3f} sec.")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def search_dense(self, query: str, top_k: int = 30) -> list[dict]:
        """Dense semantic search using BGE-M3 and FAISS."""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["score"] = float(score)
            item["dense_score"] = float(score)
            item["doc_idx"] = int(idx)
            results.append(item)

        return results

    def search_sparse(self, query: str, top_k: int = 30) -> list[dict]:
        """Sparse keyword search using BM25."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
                continue
            item = self.metadata[idx].copy()
            item["score"] = score
            item["bm25_score"] = score
            item["doc_idx"] = int(idx)
            results.append(item)

        return results

    def search(
        self,
        query: str,
        top_k: int = 30,
        hybrid: bool = USE_HYBRID_SEARCH,
    ) -> list[dict]:
        """
        Hybrid search combining Dense (FAISS) and Sparse (BM25) via Reciprocal Rank Fusion (RRF).
        Falls back to pure dense search if hybrid is False.
        """
        t0 = time.perf_counter()

        if not hybrid:
            dense_results = self.search_dense(query, top_k=top_k)
            t1 = time.perf_counter()
            print(f"Dense Search  : {t1 - t0:.3f} sec")
            return dense_results

        # Hybrid: Retrieve candidates from both systems
        candidate_k = max(top_k * 2, 50)
        dense_results = self.search_dense(query, top_k=candidate_k)
        sparse_results = self.search_sparse(query, top_k=candidate_k)

        t1 = time.perf_counter()

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        item_map = {}

        # 1. Dense ranks
        for rank, item in enumerate(dense_results, start=1):
            idx = item["doc_idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank))
            item_map[idx] = item

        # 2. Sparse ranks
        for rank, item in enumerate(sparse_results, start=1):
            idx = item["doc_idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank))
            if idx not in item_map:
                item_map[idx] = item
            else:
                item_map[idx]["bm25_score"] = item.get("bm25_score", 0.0)

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for idx in sorted_indices[:top_k]:
            item = item_map[idx]
            item["rrf_score"] = rrf_scores[idx]
            item["score"] = item.get("dense_score", item.get("score", 0.0))
            results.append(item)

        t2 = time.perf_counter()
        print(f"Hybrid Retrieval (Dense + BM25 RRF): {t2 - t0:.3f} sec")

        return results