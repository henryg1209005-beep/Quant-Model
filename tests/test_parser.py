from __future__ import annotations

import unittest

from ai_trading_engine.context_builder import SYSTEM_PROMPT
from ai_trading_engine.llm.parser import parse_decision


class TestParser(unittest.TestCase):
    def test_parse_decision_basic(self) -> None:
        raw = '{"action":"trade","direction":"LONG","confidence":0.78,"size":3,"sl_ticks":8,"tp_ticks":14,"reasoning":"test"}'
        d = parse_decision(raw)

        self.assertEqual(d.action, "trade")
        self.assertEqual(d.direction, "LONG")
        self.assertEqual(d.size, 3)
        self.assertEqual(d.sl_ticks, 8)
        self.assertEqual(d.tp_ticks, 14)

    def test_parse_decision_normalizes_direction(self) -> None:
        raw = 'Decision: {"action":" TRADE ","direction":" long ","confidence":0.72,"size":1,"sl_ticks":8,"tp_ticks":12,"reasoning":"test"}'
        d = parse_decision(raw)

        self.assertEqual(d.action, "trade")
        self.assertEqual(d.direction, "LONG")

    def test_parse_decision_forecast_fields(self) -> None:
        raw = '{"action":"hold","confidence":0.4,"size":1,"sl_ticks":0,"tp_ticks":0,"forecast_direction":" short ","forecast_confidence":0.63,"forecast_horizon_minutes":15,"reasoning":"test"}'
        d = parse_decision(raw)

        self.assertEqual(d.action, "hold")
        self.assertIsNone(d.direction)
        self.assertEqual(d.forecast_direction, "SHORT")
        self.assertEqual(d.forecast_confidence, 0.63)
        self.assertEqual(d.forecast_horizon_minutes, 15)

    def test_parse_decision_from_markdown_block(self) -> None:
        raw = """```json
{"action":"hold","confidence":0.4,"size":1,"sl_ticks":0,"tp_ticks":0,"forecast_direction":"LONG","forecast_confidence":0.6,"forecast_horizon_minutes":15,"reasoning":"ok"}
```"""
        d = parse_decision(raw)
        self.assertEqual(d.forecast_direction, "LONG")

    def test_parse_decision_from_wrapped_text(self) -> None:
        raw = 'Model output: {"action":"trade","direction":"SHORT","confidence":0.55,"size":1,"sl_ticks":6,"tp_ticks":10,"forecast_direction":"SHORT","forecast_confidence":0.55,"forecast_horizon_minutes":15,"reasoning":"ok"} end.'
        d = parse_decision(raw)
        self.assertEqual(d.action, "trade")
        self.assertEqual(d.direction, "SHORT")

    def test_parse_decision_from_key_value_block(self) -> None:
        raw = """
action: hold
confidence: 0.41
size: 1
sl_ticks: 0
tp_ticks: 0
forecast_direction: LONG
forecast_confidence: 0.62
forecast_horizon_minutes: 15
reasoning: compact repair output
"""
        d = parse_decision(raw)
        self.assertEqual(d.action, "hold")
        self.assertEqual(d.forecast_direction, "LONG")
        self.assertEqual(d.forecast_confidence, 0.62)

    def test_system_prompt_requests_compact_reasoning(self) -> None:
        self.assertIn("maximum 160 characters", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
