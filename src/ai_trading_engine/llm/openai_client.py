from __future__ import annotations

import os

from ai_trading_engine.llm.base import LlmClient


class OpenAiLlmClient(LlmClient):
    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str = "low",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._reasoning_effort = str(reasoning_effort or "").strip().lower()
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens if max_output_tokens and max_output_tokens > 0 else None
        self._api_key = os.getenv("OPENAI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package not installed; install with `pip install -e .[openai]`") from exc

        self._client = OpenAI(api_key=self._api_key)

    def decide(self, system_prompt: str, context: str) -> str:
        request = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
        }
        if self._reasoning_effort and self._reasoning_effort != "none":
            request["reasoning"] = {"effort": self._reasoning_effort}
        if self._temperature is not None:
            request["temperature"] = float(self._temperature)
        if self._max_output_tokens is not None:
            request["max_output_tokens"] = int(self._max_output_tokens)

        try:
            response = self._client.responses.create(**request)
        except TypeError:
            request.pop("reasoning", None)
            response = self._client.responses.create(**request)
        return response.output_text
