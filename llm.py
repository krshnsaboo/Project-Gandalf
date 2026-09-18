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

    def generate_with_cost(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = 2,
    ) -> tuple[str, dict]:
        """
        Generates completions from OpenAI with retry logic and calculates token usage & cost (USD/INR).
        """
        self._ensure_client()

        USD_PER_INPUT_TOKEN = 0.15 / 1_000_000   # $0.15 per 1M tokens for gpt-4o-mini
        USD_PER_OUTPUT_TOKEN = 0.60 / 1_000_000  # $0.60 per 1M tokens for gpt-4o-mini
        USD_TO_INR = 84.0                        # Standard exchange rate

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
                text = response.choices[0].message.content.strip()

                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0
                    total_tokens = response.usage.total_tokens or (prompt_tokens + completion_tokens)

                cost_usd = (prompt_tokens * USD_PER_INPUT_TOKEN) + (completion_tokens * USD_PER_OUTPUT_TOKEN)
                cost_inr = cost_usd * USD_TO_INR

                cost_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost_usd,
                    "cost_inr": cost_inr,
                }

                return text, cost_info

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

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = 2,
    ) -> str:
        text, _ = self.generate_with_cost(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        return text