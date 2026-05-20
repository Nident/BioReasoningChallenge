from __future__ import annotations

from typing import Any

from openai import OpenAI


class Model:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 32,
        top_p: float = 1.0,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.client = client or self._load_client()

    def _load_client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("api_key is required")

        if self.base_url:
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        return OpenAI(api_key=self.api_key)

    def request(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content:
            raise ValueError("model returned empty response")
        return content
