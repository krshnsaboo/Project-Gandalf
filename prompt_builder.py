from typing import List, Tuple

SYSTEM_PROMPT = (
    "You are an expert navigation assistant for Striver's A2Z DSA Course. "
    "Do NOT explain algorithms or answer questions. "
    "ONLY identify and return the 1 single most relevant lecture timestamp from the provided segments."
)


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

        user_prompt = f"Question:\n{query}\n\nSegments:\n"

        for i, ctx in enumerate(contexts, start=1):
            start = format_timestamp(ctx.get("start", 0))
            user_prompt += (
                f"[{i}] {ctx.get('lecture_title', '')}\n"
                f"Timestamp: {start}\n"
                f"Watch: {ctx.get('timestamp_url', '')}\n"
                f"Transcript: {ctx.get('text', '').strip()}\n\n"
            )

        user_prompt += (
            "Select the 1 best matching timestamp from the segments above.\n"
            "Format EXACTLY:\n\n"
            "Lecture: <Lecture Title>\n"
            "Timestamp: <MM:SS>\n"
            "Watch: <Timestamp URL>\n"
        )

        return SYSTEM_PROMPT, user_prompt