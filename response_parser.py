import re


class ResponseParser:

    @staticmethod
    def parse(response: str):

        recommendations = []

        pattern = re.compile(
            r"Lecture:\s*(.*?)\n"
            r"Timestamp:\s*(.*?)\n"
            r"Watch:\s*(.*?)(?=\nLecture:|\Z)",
            re.DOTALL
        )

        matches = pattern.findall(response)

        for lecture, timestamp, url in matches:

            recommendations.append(
                {
                    "lecture": lecture.strip(),
                    "timestamp": timestamp.strip(),
                    "url": url.strip()
                }
            )

        return recommendations