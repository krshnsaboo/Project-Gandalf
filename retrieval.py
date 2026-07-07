import pickle
import faiss
import time
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(
        self,
        embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
        index_path="lecture_embeddings/faiss_index.bin",
        model_name="BAAI/bge-m3",
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

    def search(self, query, top_k=5):

        t0 = time.perf_counter()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        t1 = time.perf_counter()

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        t2 = time.perf_counter()

        results = []

        for score, idx in zip(scores[0], indices[0]):

            item = self.metadata[idx].copy()
            item["score"] = float(score)

            results.append(item)
        

        print(f"Embedding : {t1 - t0:.3f} sec")
        print(f"FAISS     : {t2 - t1:.3f} sec")
        return results