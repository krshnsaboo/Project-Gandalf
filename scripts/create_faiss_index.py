import json
import pickle
from datetime import datetime

import faiss
import numpy as np

EMBEDDINGS_PATH = "lecture_embeddings/all_lecture_embeddings.pkl"
INDEX_PATH = "lecture_embeddings/faiss_index.bin"
INFO_PATH = "lecture_embeddings/faiss_info.json"

MODEL_NAME = "BAAI/bge-m3"


def load_embeddings():
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = []

    for lecture in data:
        embeddings.append(lecture["embeddings"])

    embeddings = np.vstack(embeddings).astype(np.float32)

    return embeddings


def save_index_info(embeddings):
    info = {
        "model": MODEL_NAME,
        "dimension": int(embeddings.shape[1]),
        "num_vectors": int(embeddings.shape[0]),
        "normalized": True,
        "index_type": "IndexFlatIP",
        "embedding_dtype": str(embeddings.dtype),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(f"Metadata saved to: {INFO_PATH}")


def main():

    print("Loading embeddings...")
    embeddings = load_embeddings()

    print(f"Loaded {len(embeddings)} embeddings.")

    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_PATH)

    print("Saving index metadata...")
    save_index_info(embeddings)

    print("\nDone!")
    print(f"Index Path      : {INDEX_PATH}")
    print(f"Metadata Path   : {INFO_PATH}")
    print(f"Embedding Model : {MODEL_NAME}")
    print(f"Dimension       : {dimension}")
    print(f"Vectors Indexed : {index.ntotal}")


if __name__ == "__main__":
    main()