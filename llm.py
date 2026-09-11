import os
import time
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError

from config import get_openai_api_key, OPENAI_MODEL, TEMPERATURE, MAX_TOKENS


class LLM:

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_openai_api_key(required=False)
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def _ensure_client(self):
        if self.client is None:
            self.api_key = get_openai_api_key(required=True)
            self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = 2,
    ) -> str:
        """
        Generates completions from OpenAI with retry logic and error resilience.
        """
        self._ensure_client()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content.strip()

            except (RateLimitError, APIConnectionError) as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise e

            except AuthenticationError as e:
                raise ValueError(
                    "Invalid OpenAI API Key. Please verify your OPENAI_API_KEY in .env or Streamlit Secrets."
                ) from e

            except APIError as e:
                last_error = e
                raise e

        raise last_error