import json
from retrieval import Retriever

# ----------------------------
# Configuration
# ----------------------------
EVAL_FILE = "evaluation_queries.json"
TOP_K = 5

# ----------------------------
# Load Retriever
# ----------------------------
retriever = Retriever(
    embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
    index_path="lecture_embeddings/faiss_index.bin",
)

# ----------------------------
# Load Evaluation Queries
# ----------------------------
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)

total_queries = len(queries)

recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0

mrr = 0.0

print("\n" + "=" * 90)
print("RAG RETRIEVAL EVALUATION")
print("=" * 90)

for idx, sample in enumerate(queries, start=1):

    query = sample["query"]
    expected = sample["expected_lecture"]

    results = retriever.search(query, top_k=TOP_K)

    predicted = [r["lecture_id"] for r in results]

    rank = None

    for i, lecture in enumerate(predicted):
        if lecture == expected:
            rank = i + 1
            break

    hit1 = rank == 1
    hit5 = rank is not None and rank <= 5
    hit10 = rank is not None and rank <= 10

    if hit1:
        recall_at_1 += 1

    if hit5:
        recall_at_5 += 1

    if hit10:
        recall_at_10 += 1

    if rank is not None:
        mrr += 1.0 / rank

    # ------------------------
    # Print Query Result
    # ------------------------

    print(f"\nQuery {idx}/{total_queries}")
    print("-" * 90)
    print(f"Query            : {query}")
    print(f"Expected Lecture : {expected}")

    if rank is None:
        print("Found At         : NOT FOUND")
    else:
        print(f"Found At Rank    : {rank}")

    print("\nTop Results:")

    for i, r in enumerate(results, start=1):

        mark = "✅" if r["lecture_id"] == expected else " "

        print(
            f"{i:2d}. {mark} "
            f"{r['lecture_id']} | "
            f"{r['lecture_title']} | "
            f"Score: {r['score']:.4f}"
        )

# ----------------------------
# Final Metrics
# ----------------------------

recall1 = recall_at_1 / total_queries
recall5 = recall_at_5 / total_queries
recall10 = recall_at_10 / total_queries
mrr /= total_queries

print("\n" + "=" * 90)
print("FINAL RESULTS")
print("=" * 90)

print(f"Total Queries : {total_queries}")

print(f"\nRecall@1      : {recall1:.2%}")
print(f"Recall@5      : {recall5:.2%}")
print(f"Recall@10     : {recall10:.2%}")

print(f"\nMRR           : {mrr:.4f}")

print("=" * 90)