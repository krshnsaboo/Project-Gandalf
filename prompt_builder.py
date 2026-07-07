from typing import List, Tuple


SYSTEM_PROMPT = """
You are an expert navigation assistant for Striver's A2Z DSA Course.

Your job is NOT to answer the user's DSA question.

Your job is ONLY to guide the user to the most relevant timestamp(s) from the retrieved lecture transcripts.

Instructions:
- Read all retrieved contexts carefully.
- Identify the 1 timestamps that best answer the user's question.
- If multiple timestamps are useful, list them in order of importance.
- Do NOT explain the algorithm.
- Do NOT summarize the transcript.
- Do NOT answer the user's question.
- If the retrieved contexts do not contain the answer, clearly say that.
- Respond naturally, as if guiding someone to the correct part of the lecture.
"""


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"

    return f"{minutes:02}:{secs:02}"


class PromptBuilder:

    @staticmethod
    def build(
        query: str,
        contexts: List[dict]
    ) -> Tuple[str, str]:

        user_prompt = f"User Question:\n{query}\n\n"

        user_prompt += "Retrieved Lecture Segments:\n\n"

        for i, ctx in enumerate(contexts, start=1):

            start = format_timestamp(ctx["start"])
            end = format_timestamp(ctx["end"])

            user_prompt += (
                f"Candidate {i}\n"
                f"Lecture: {ctx['lecture_title']}\n"
                f"Timestamp: {start}\n"
                f"URL: {ctx['timestamp_url']}\n\n"
                f"Transcript:\n"
                f"{ctx['text']}\n\n"
                f"{'-' * 80}\n\n"
            )

        user_prompt += """
Based ONLY on the retrieved lecture segments above, recommend the most relevant lecture timestamp(s).

Rules:
- Do NOT answer the user's question.
- Do NOT explain the algorithm.
- Recommend top most 1 timestamps.
- Use ONLY the lecture titles, timestamps and URLs provided above.
- Never invent information.
- If the answer cannot be found, clearly state that.

Return your response EXACTLY in the following format:

Lecture: <Lecture Title>
Timestamp: <MM:SS>
Watch: <Timestamp URL>

Lecture: <Lecture Title>
Timestamp: <MM:SS>
Watch: <Timestamp URL>

Do not add any additional explanation before or after the recommendations.
"""

        return SYSTEM_PROMPT, user_prompt