import re
from typing import List, Dict, Any


class ResponseParser:

    @staticmethod
    def parse(response: str) -> List[Dict[str, Any]]:
        """
        Parses LLM response into structured recommendation dictionaries.
        Robust to standard text, markdown formatting (**Lecture:**), and slight variations.
        Returns a list of dicts with keys: 'title', 'lecture', 'timestamp', 'url', 'description'.
        """
        if not response or not response.strip():
            return []

        recommendations = []

        # Primary pattern: supports both plain and markdown bold labels
        pattern = re.compile(
            r"(?:\*\*Lecture:\*\*|Lecture:)\s*(.*?)\n"
            r"(?:\*\*Timestamp:\*\*|Timestamp:)\s*(.*?)\n"
            r"(?:\*\*Watch:\*\*|Watch:)\s*(https?://[^\s\n]+)",
            re.IGNORECASE
        )

        matches = pattern.findall(response)

        for lecture, timestamp, url in matches:
            cleaned_title = lecture.strip()
            cleaned_time = timestamp.strip()
            cleaned_url = url.strip().rstrip(").,")

            recommendations.append({
                "title": cleaned_title,
                "lecture": cleaned_title,
                "timestamp": cleaned_time,
                "url": cleaned_url,
                "description": "",
            })

        # Fallback parser if primary structured pattern missed results
        if not recommendations:
            blocks = [b.strip() for b in response.strip().split("\n\n") if b.strip()]
            for block in blocks:
                title = ""
                timestamp = ""
                url = ""
                desc_lines = []

                for line in block.splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    # Check for URL
                    url_match = re.search(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s\)\>]+", line)
                    if url_match:
                        url = url_match.group(0).rstrip(").,")

                    # Check for timestamp
                    time_match = re.search(r"(?:timestamp:\s*|\b)(\d{1,2}:\d{2}(?::\d{2})?)\b", line, re.IGNORECASE)
                    if time_match and not timestamp:
                        timestamp = time_match.group(1)

                    # Check for lecture title
                    if re.match(r"^(?:\*\*Lecture:\*\*|Lecture:)\s*", line, re.IGNORECASE):
                        title = re.sub(r"^(?:\*\*Lecture:\*\*|Lecture:)\s*", "", line, flags=re.IGNORECASE).strip()
                    elif not title and not line.lower().startswith(("watch", "http", "timestamp", "-", "*")):
                        title = line
                    else:
                        desc_lines.append(line)

                if url or title:
                    recommendations.append({
                        "title": title or "Lecture Recommendation",
                        "lecture": title or "Lecture Recommendation",
                        "timestamp": timestamp,
                        "url": url,
                        "description": " ".join(desc_lines),
                    })

        return recommendations