import re


def normalize_query(query: str) -> str:
    """
    Normalizes user input queries.
    If the query is a LeetCode problem URL (e.g. https://leetcode.com/problems/add-two-numbers/),
    extracts and cleans the problem title (e.g. 'Add Two Numbers').
    Otherwise returns the cleaned query string.
    """
    if not query:
        return ""

    trimmed = query.strip()

    # Detect LeetCode problem URLs (com or cn domain)
    leetcode_match = re.search(r"leetcode\.(?:com|cn)/problems/([^/?#]+)", trimmed, re.IGNORECASE)
    if leetcode_match:
        slug = leetcode_match.group(1).strip()
        # Convert kebab-case slug (e.g., 'add-two-numbers') to title case ('Add Two Numbers')
        cleaned_slug = slug.replace("-", " ").replace("_", " ")
        cleaned_title = " ".join(word.capitalize() for word in cleaned_slug.split())
        return cleaned_title

    return trimmed
