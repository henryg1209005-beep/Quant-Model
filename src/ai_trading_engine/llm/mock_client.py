from __future__ import annotations

import json
import random

from ai_trading_engine.llm.base import LlmClient


class MockLlmClient(LlmClient):
    """Simple deterministic-ish LLM replacement for local testing."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def decide(self, system_prompt: str, context: str) -> str:
        del system_prompt
        ctx = context.upper()

        bull = "STRONG_BULL" in ctx or " BULL" in ctx
        bear = "STRONG_BEAR" in ctx or " BEAR" in ctx
        extreme_vol = "EXTREME" in ctx

        if extreme_vol and self._rng.random() < 0.5:
            action = "hold"
            direction = "LONG"
            confidence = 0.38
            size = 1
            sl = 10
            tp = 12
            reason = "Volatility is extreme; preserve capital."
        elif bull and not bear:
            action = "trade"
            direction = "LONG"
            confidence = 0.64 + self._rng.random() * 0.2
            size = 2 + int(self._rng.random() * 4)
            sl = 8
            tp = 14
            reason = "Bullish context with supportive trend and momentum."
        elif bear and not bull:
            action = "trade"
            direction = "SHORT"
            confidence = 0.64 + self._rng.random() * 0.2
            size = 2 + int(self._rng.random() * 4)
            sl = 8
            tp = 14
            reason = "Bearish context with downside continuation risk."
        else:
            action = "hold"
            direction = "LONG"
            confidence = 0.45
            size = 1
            sl = 8
            tp = 10
            reason = "Mixed signals; no clean edge."

        return json.dumps(
            {
                "action": action,
                "direction": direction,
                "confidence": round(min(max(confidence, 0.0), 1.0), 2),
                "size": size,
                "sl_ticks": sl,
                "tp_ticks": tp,
                "reasoning": reason,
            }
        )
