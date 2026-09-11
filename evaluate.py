import sys
import json

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from retrieval import Retriever
from query_normalizer import normalize_query

# ----------------------------
# Configuration
# ----------------------------
EVAL_FILE = "evaluation_queries.json"
TOP_K = 10

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
print(f"RAG RETRIEVAL EVALUATION (TOP_K = {TOP_K})")
print("=" * 90)

for idx, sample in enumerate(queries, start=1):

    raw_query = sample["query"]
    query = normalize_query(raw_query)
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
        print("Found At         : NOT FOUND (in top 10)")
    else:
        print(f"Found At Rank    : {rank}")

    print("\nTop Results:")
    for i, r in enumerate(results[:5], start=1):
        mark = "[HIT]" if r["lecture_id"] == expected else "     "
        score_val = r.get("rrf_score", r.get("score", 0.0))
        print(
            f"{i:2d}. {mark} "
            f"{r['lecture_id']} | "
            f"{r['lecture_title']} | "
            f"Score: {score_val:.4f}"
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