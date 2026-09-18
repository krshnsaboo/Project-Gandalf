import os
import sys
import json
import contextlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

from retrieval import Retriever
from reranker import Reranker
from query_normalizer import normalize_query
from config import RERANKER_MODEL

EVAL_FILE = "evaluation_queries.json"
OUTPUT_FILE = "evaluation_results.json"
TOP_K = 10

limit_val = None
model_name = RERANKER_MODEL

for arg in sys.argv:
    if arg.startswith("--limit="):
        try:
            limit_val = int(arg.split("=")[1])
        except Exception:
            pass
    elif arg == "--limit":
        try:
            idx = sys.argv.index("--limit")
            if idx + 1 < len(sys.argv):
                limit_val = int(sys.argv[idx + 1])
        except Exception:
            pass
    elif arg.startswith("--model="):
        model_name = arg.split("=")[1]
    elif arg == "--model":
        try:
            idx = sys.argv.index("--model")
            if idx + 1 < len(sys.argv):
                model_name = sys.argv[idx + 1]
        except Exception:
            pass


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


with suppress_stdout():
    retriever = Retriever(
        embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
        index_path="lecture_embeddings/faiss_index.bin",
    )
    reranker = Reranker(model_name=model_name)

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)

if limit_val is not None and limit_val > 0:
    queries = queries[:limit_val]

total_queries = len(queries)

recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0
mrr = 0.0
exact_chunk_matches = 0
within_5_mins = 0
within_10_mins = 0
total_time_diff = 0.0
hit1_count = 0

print(f"Evaluating {total_queries} queries with {model_name}...", flush=True)

for idx, sample in enumerate(queries, start=1):
    raw_query = sample["query"]
    query = normalize_query(raw_query)
    expected_lecture = sample["expected_lecture"]
    expected_sec = sample.get("expected_timestamp_sec", 0)
    expected_chunk = sample.get("expected_chunk_id", "")

    with suppress_stdout():
        candidates = retriever.search(query, top_k=20)
        results = reranker.rerank(query, candidates, top_k=TOP_K)

    predicted_lectures = [r["lecture_id"] for r in results]

    rank = None
    for i, lid in enumerate(predicted_lectures):
        if lid == expected_lecture:
            rank = i + 1
            break

    if rank == 1:
        recall_at_1 += 1
        hit1_count += 1
        top_result = results[0]
        if top_result.get("chunk_id") == expected_chunk:
            exact_chunk_matches += 1
        time_diff = abs(top_result.get("start", 0) - expected_sec)
        if time_diff <= 300:
            within_5_mins += 1
        if time_diff <= 600:
            within_10_mins += 1
        total_time_diff += time_diff

    if rank is not None and rank <= 5:
        recall_at_5 += 1
    if rank is not None and rank <= 10:
        recall_at_10 += 1
    if rank is not None:
        mrr += 1.0 / rank

    if idx % 50 == 0 or idx == total_queries:
        print(f"Processed {idx}/{total_queries} queries...", flush=True)

r1_pct = recall_at_1 / total_queries
r5_pct = recall_at_5 / total_queries
r10_pct = recall_at_10 / total_queries
mrr_val = mrr / total_queries
chunk_pct = exact_chunk_matches / total_queries
w5m_pct = within_5_mins / total_queries
w10m_pct = within_10_mins / total_queries
mae_sec = (total_time_diff / hit1_count) if hit1_count > 0 else 0.0

final_response = {
    "total_queries": total_queries,
    "model": model_name,
    "recall_at_1": r1_pct,
    "recall_at_5": r5_pct,
    "recall_at_10": r10_pct,
    "mrr": mrr_val,
    "exact_chunk_match": chunk_pct,
    "within_5_minutes": w5m_pct,
    "within_10_minutes": w10m_pct,
    "timestamp_mae_seconds": mae_sec,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_response, f, indent=2)

print("\n" + "=" * 90)
print(f"OFFICIAL BENCHMARK EVALUATION RESULTS (TOTAL QUERIES: {total_queries})")
print(f"Model: {model_name}")
print("=" * 90)
print(f"{'Metric':<36} | {'Result':<40}")
print("-" * 36 + "-+-" + "-" * 40)
print(f"{'Recall@1':<36} | {r1_pct:.2%} ({recall_at_1}/{total_queries})")
print(f"{'Recall@5':<36} | {r5_pct:.2%} ({recall_at_5}/{total_queries})")
print(f"{'Recall@10':<36} | {r10_pct:.2%} ({recall_at_10}/{total_queries})")
print(f"{'MRR (Mean Reciprocal Rank)':<36} | {mrr_val:.4f}")
print("-" * 36 + "-+-" + "-" * 40)
print(f"{'Exact Chunk Match':<36} | {chunk_pct:.2%} ({exact_chunk_matches}/{total_queries})")
print(f"{'Timestamp within 5 minutes':<36} | {w5m_pct:.2%} ({within_5_mins}/{total_queries})")
print(f"{'Timestamp within 10 minutes':<36} | {w10m_pct:.2%} ({within_10_mins}/{total_queries})")
print(f"{'Timestamp Mean Absolute Error (MAE)':<36} | {mae_sec:.1f} seconds")
print("=" * 90)
print(f"Results saved to {OUTPUT_FILE}\n")