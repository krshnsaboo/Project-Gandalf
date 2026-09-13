import re

KNOWN_ACRONYMS = {
    "lru": "LRU",
    "lfu": "LFU",
    "bst": "BST",
    "dfs": "DFS",
    "bfs": "BFS",
    "dp": "DP",
    "dsa": "DSA",
    "mst": "MST",
    "pq": "PQ",
    "ii": "II",
    "iii": "III",
    "iv": "IV",
    "vi": "VI",
    "vii": "VII",
}

EXPANSIONS = [
    (re.compile(r"\bpq\b", re.IGNORECASE), "Priority Queue"),
    (re.compile(r"\bdp\b", re.IGNORECASE), "Dynamic Programming"),
    (re.compile(r"\bbst\b", re.IGNORECASE), "Binary Search Tree"),
    (re.compile(r"\blru\b", re.IGNORECASE), "LRU Cache"),
    (re.compile(r"\blfu\b", re.IGNORECASE), "LFU Cache"),
    (re.compile(r"\bdfs\b", re.IGNORECASE), "Depth First Search"),
    (re.compile(r"\bbfs\b", re.IGNORECASE), "Breadth First Search"),
    (re.compile(r"\bmst\b", re.IGNORECASE), "Minimum Spanning Tree"),
    (re.compile(r"\bll\b", re.IGNORECASE), "Linked List"),
    (re.compile(r"\btc\b", re.IGNORECASE), "Time Complexity"),
    (re.compile(r"\bsc\b", re.IGNORECASE), "Space Complexity"),
    (re.compile(r"\bopt\b", re.IGNORECASE), "Optimal"),
    (re.compile(r"\bsol\b", re.IGNORECASE), "Solution"),
]


def _format_word(word: str) -> str:
    lower = word.lower()
    if lower in KNOWN_ACRONYMS:
        return KNOWN_ACRONYMS[lower]
    return word.capitalize()


def normalize_query(query: str) -> str:
    if not query:
        return ""

    trimmed = query.strip()

    leetcode_match = re.search(r"leetcode\.(?:com|cn)/problems/([^/?#]+)", trimmed, re.IGNORECASE)
    if leetcode_match:
        slug = leetcode_match.group(1).strip()
        cleaned_slug = slug.replace("-", " ").replace("_", " ")
        cleaned_title = " ".join(_format_word(word) for word in cleaned_slug.split())
        return cleaned_title

    return trimmed


def enhance_query(query: str) -> str:
    normalized = normalize_query(query)
    if not normalized:
        return ""

    expanded = normalized
    for pattern, replacement in EXPANSIONS:
        expanded = pattern.sub(replacement, expanded)

    expanded = re.sub(r"\s+", " ", expanded).strip()

    question_starters = ("which", "where", "how", "what", "when", "can", "is", "find")
    if any(expanded.lower().startswith(q) for q in question_starters):
        return expanded

    return f"Which lecture and timestamp covers {expanded}?"
