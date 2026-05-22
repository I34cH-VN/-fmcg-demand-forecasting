from __future__ import annotations

import os

from src.agent.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional openai package to use OpenAILLM.") from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")

        client = OpenAI()
        response = client.responses.create(model=self.model, input=prompt)
        return response.output_text
