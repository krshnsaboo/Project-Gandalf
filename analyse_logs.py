import json
from pathlib import Path
from collections import Counter
from statistics import mean, median


LOG_FILE = Path("logs/search_logs.jsonl")


def load_logs():
    if not LOG_FILE.exists():
        print("Log file not found:", LOG_FILE)
        return []

    logs = []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return logs


def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_section(title):
    print(f"\n{title}")
    print("-" * 90)


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_median(values):
    return median(values) if values else 0.0


logs = load_logs()

if not logs:
    print("No logs found.")
    exit()

total_searches = len(logs)

queries = [x["query"].strip() for x in logs]

query_counter = Counter(queries)
lecture_counter = Counter(
    x["selected_result"]["lecture_title"] for x in logs
)

retrieval_times = [
    x["latency"]["retrieval_sec"] for x in logs
]

rerank_times = [
    x["latency"]["reranking_sec"] for x in logs
]

llm_times = [
    x["latency"]["llm_sec"] for x in logs
]

total_times = [
    x["latency"]["total_sec"] for x in logs
]

faiss_scores = [
    x["selected_result"]["faiss_score"]
    for x in logs
]

rerank_scores = [
    x["selected_result"]["rerank_score"]
    for x in logs
]


print_header("PROJECT GANDALF - SEARCH ANALYTICS")


# ==========================================================
# GENERAL
# ==========================================================

unique_queries = len(query_counter)
duplicate_searches = total_searches - unique_queries

print_section("GENERAL")

print(f"{'Total Searches':30}: {total_searches}")
print(f"{'Unique Queries':30}: {unique_queries}")
print(
    f"{'Duplicate Searches':30}: "
    f"{duplicate_searches} "
    f"({duplicate_searches / total_searches * 100:.2f}%)"
)


# ==========================================================
# LATENCY
# ==========================================================

print_section("LATENCY")

print(f"{'Average Retrieval':30}: {safe_mean(retrieval_times):.3f} sec")
print(f"{'Average Reranking':30}: {safe_mean(rerank_times):.3f} sec")
print(f"{'Average LLM':30}: {safe_mean(llm_times):.3f} sec")
print(f"{'Average Total':30}: {safe_mean(total_times):.3f} sec")

print()

print(f"{'Minimum Total':30}: {min(total_times):.3f} sec")
print(f"{'Median Total':30}: {safe_median(total_times):.3f} sec")
print(f"{'Maximum Total':30}: {max(total_times):.3f} sec")


# ==========================================================
# TIME DISTRIBUTION
# ==========================================================

print_section("TIME DISTRIBUTION")

avg_total = safe_mean(total_times)

retrieval_pct = safe_mean(retrieval_times) / avg_total * 100
rerank_pct = safe_mean(rerank_times) / avg_total * 100
llm_pct = safe_mean(llm_times) / avg_total * 100

print(f"{'Retrieval':30}: {retrieval_pct:.1f}%")
print(f"{'Reranking':30}: {rerank_pct:.1f}%")
print(f"{'LLM':30}: {llm_pct:.1f}%")


# ==========================================================
# RETRIEVAL QUALITY
# ==========================================================

print_section("RETRIEVAL QUALITY")

print(f"{'Average FAISS Score':30}: {safe_mean(faiss_scores):.4f}")
print(f"{'Average Rerank Score':30}: {safe_mean(rerank_scores):.4f}")


# ==========================================================
# LOWEST RERANK SCORES
# ==========================================================

print_section("LOWEST RERANK SCORE SEARCHES")

lowest = sorted(
    logs,
    key=lambda x: x["selected_result"]["rerank_score"]
)[:5]

for i, log in enumerate(lowest, start=1):

    print(f"\n{i}. Query : {log['query']}")
    print(
        f"   Score : "
        f"{log['selected_result']['rerank_score']:.4f}"
    )


# ==========================================================
# TOP SEARCHES
# ==========================================================

print_section("TOP SEARCH QUERIES")

for i, (query, count) in enumerate(
        query_counter.most_common(10),
        start=1):

    print(f"{i:>2}. ({count:>3}) {query}")


# ==========================================================
# TOPIC DISTRIBUTION
# ==========================================================

print_section("TOPIC DISTRIBUTION")

topic_counter = Counter()

keywords = [
    "Array",
    "Binary Search",
    "Sliding Window",
    "Two Pointer",
    "Linked List",
    "Stack",
    "Queue",
    "Tree",
    "Binary Tree",
    "BST",
    "Graph",
    "DFS",
    "BFS",
    "Trie",
    "Heap",
    "Segment Tree",
    "Fenwick",
    "DP",
    "Dynamic Programming",
    "Greedy",
    "Recursion",
    "Backtracking",
    "Bit",
    "Math",
    "String",
    "Hash",
    "Disjoint",
    "Union Find",
    "MST",
    "Shortest Path",
]

for lecture in lecture_counter.elements():

    found = False

    lecture_lower = lecture.lower()

    for keyword in keywords:

        if keyword.lower() in lecture_lower:
            topic_counter[keyword] += 1
            found = True
            break

    if not found:
        topic_counter["Others"] += 1


for topic, count in topic_counter.most_common():

    print(f"{topic:30} ({count})")


# ==========================================================
# MOST RETRIEVED LECTURES
# ==========================================================

print_section("MOST RETRIEVED LECTURES")

for i, (lecture, count) in enumerate(
        lecture_counter.most_common(10),
        start=1):

    print(f"{i:>2}. ({count:>3}) {lecture}")


print("\n" + "=" * 90)
print("End of Report")
print("=" * 90)