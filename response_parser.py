import re
from typing import List, Dict, Any


class ResponseParser:

    @staticmethod
    def parse(response: str) -> List[Dict[str, Any]]:
        if not response or not response.strip():
            return []

        recommendations = []

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

                    url_match = re.search(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s\)\>]+", line)
                    if url_match:
                        url = url_match.group(0).rstrip(").,")

                    time_match = re.search(r"(?:timestamp:\s*|\b)(\d{1,2}:\d{2}(?::\d{2})?)\b", line, re.IGNORECASE)
                    if time_match and not timestamp:
                        timestamp = time_match.group(1)

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