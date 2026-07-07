from retrieval import Retriever
from reranker import Reranker

retriever = Retriever(
    embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
    index_path="lecture_embeddings/faiss_index.bin",
)

reranker = Reranker()

while True:
    query = input("\nEnter query (or 'exit'): ").strip()

    if query.lower() == "exit":
        break

    candidates = retriever.search(query, top_k=30)

    results = reranker.rerank(
        query,
        candidates,
        top_k=5
    )

    print("\n" + "=" * 80)

    if not results:
        print("No relevant results found.")
        continue

    for result in results:
        print(f"FAISS Score    : {result['score']:.4f}")
        print(f"Rerank Score   : {result['rerank_score']:.4f}")
        print(f"Lecture : {result['lecture_title']}")
        print(f"Chunk   : {result['chunk_id']}")
        print(f"Time    : {result['start']:.2f}s - {result['end']:.2f}s")
        print(f"URL     : {result['timestamp_url']}")
        print("Text:")
        print(result["text"])
        print("=" * 80)