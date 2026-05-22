from __future__ import annotations

from src.agent.base_llm import BaseLLM


class MockLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "recommend" in prompt_lower or "next action" in prompt_lower:
            return "Prioritize data quality fixes, monitor high-error segments, and review demand drivers before production rollout."
        if "forecast" in prompt_lower or "metric" in prompt_lower:
            return "Forecast quality should be judged with WMAPE, MAE, RMSE, and bias, with special attention to systematic over- or under-forecasting."
        return "The dataset and forecast outputs were reviewed using deterministic checks and portfolio-safe mock analysis."
