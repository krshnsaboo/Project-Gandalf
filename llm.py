from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL


class LLM:

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content.strip()