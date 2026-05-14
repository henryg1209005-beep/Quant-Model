from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ai_trading_engine.llm.base import LlmClient


class GeminiLlmClient(LlmClient):
    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = None,
        timeout_seconds: int = 30,
        max_output_tokens: int | None = None,
        max_retries: int = 1,
    ) -> None:
        self._model = (model or "gemini-1.5-flash-8b").strip()
        self._temperature = temperature
        self._timeout_seconds = max(1, int(timeout_seconds))
        self._max_output_tokens = max_output_tokens if max_output_tokens and max_output_tokens > 0 else None
        self._max_retries = max(1, int(max_retries))
        self._api_key = os.getenv("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

    def decide(self, system_prompt: str, context: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self._model, safe='')}:generateContent?key={quote(self._api_key, safe='')}"
        )
        generation_config: dict[str, object] = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "action": {"type": "STRING", "enum": ["trade", "hold"]},
                    "direction": {"type": "STRING", "enum": ["LONG", "SHORT"]},
                    "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                    "size": {"type": "INTEGER", "minimum": 1, "maximum": 10},
                    "sl_ticks": {"type": "INTEGER", "minimum": 0},
                    "tp_ticks": {"type": "INTEGER", "minimum": 0},
                    "forecast_direction": {"type": "STRING", "enum": ["LONG", "SHORT"]},
                    "forecast_confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                    "forecast_horizon_minutes": {"type": "INTEGER", "minimum": 1, "maximum": 390},
                    "reasoning": {"type": "STRING"},
                },
                "required": [
                    "action",
                    "confidence",
                    "size",
                    "sl_ticks",
                    "tp_ticks",
                    "forecast_direction",
                    "forecast_confidence",
                    "forecast_horizon_minutes",
                    "reasoning",
                ],
            },
        }
        if self._temperature is not None:
            generation_config["temperature"] = float(self._temperature)
        if self._max_output_tokens is not None:
            generation_config["maxOutputTokens"] = int(self._max_output_tokens)

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": context}]}],
            "generationConfig": generation_config,
        }
        body = json.dumps(payload).encode("utf-8")
        data = None
        last_error: Exception | None = None
        max_attempts = self._max_retries
        for attempt in range(max_attempts):
            req = Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=self._timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                last_error = RuntimeError(
                    f"Gemini HTTP {exc.code}: {body[:400]}"
                )
                if exc.code not in {429, 500, 502, 503, 504} or attempt == max_attempts - 1:
                    raise last_error
                time.sleep(0.5 * (attempt + 1))
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))

        if data is None:
            raise RuntimeError(f"Gemini request failed: {last_error}")

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        text = ""
        finish_reasons: list[str] = []
        for candidate in candidates:
            finish_reason = str(candidate.get("finishReason") or "").strip()
            if finish_reason:
                finish_reasons.append(finish_reason)
            parts = ((candidate.get("content") or {}).get("parts") or [])
            candidate_text = "".join(str(part.get("text") or "") for part in parts).strip()
            if candidate_text:
                text = candidate_text
                break
        if not text:
            reason_text = ",".join(finish_reasons) if finish_reasons else "unknown"
            raise RuntimeError(f"Gemini returned empty content: finish_reason={reason_text} payload={data}")
        # A structurally valid JSON response is always longer than ~20 chars.
        # MAX_TOKENS means the output was hard-cut mid-generation.
        if len(text) < 20 and "MAX_TOKENS" in finish_reasons:
            raise RuntimeError(
                f"Gemini response truncated by token limit ({len(text)} chars): {text!r}"
            )
        return text
