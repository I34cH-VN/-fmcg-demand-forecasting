from __future__ import annotations

import os

from src.agent.base_llm import BaseLLM


class ExternalLLM(BaseLLM):
    """Generic client for chat-completions-compatible LLM APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional openai package to use ExternalLLM.") from exc

        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not set.")
        if not self.model:
            raise RuntimeError("LLM_MODEL is not set.")

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
