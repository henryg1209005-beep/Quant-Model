from __future__ import annotations

from datetime import datetime, timezone
import unittest

from ai_trading_engine.adapters.alpaca_adapter import AlpacaMarketDataAdapter


class FakeAlpacaAdapter(AlpacaMarketDataAdapter):
    def __init__(self, payloads: dict[str, dict]) -> None:
        super().__init__("key", "secret")
        self.payloads = payloads
        self.requests: list[tuple[str, dict[str, str]]] = []

    def _request_json(self, path: str, params: dict[str, str]) -> dict:
        self.requests.append((path, params))
        return self.payloads[path]


class TestAlpacaAdapter(unittest.TestCase):
    def test_historical_bars_skips_invalid_rows_and_sorts(self) -> None:
        adapter = FakeAlpacaAdapter(
            {
                "/v2/stocks/SPY/bars": {
                    "bars": [
                        {"t": "2026-04-29T13:10:00Z", "o": 101, "h": 102, "l": 100, "c": 101.5, "v": 10},
                        {"t": "bad", "o": 101, "h": 102, "l": 100, "c": 101.5, "v": 10},
                        {"t": "2026-04-29T13:05:00Z", "o": 100, "h": 99, "l": 98, "c": 100.5, "v": 10},
                        {"t": "2026-04-29T13:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 10},
                    ]
                }
            }
        )

        bars = adapter.get_historical_bars(" spy ", 5, 10, datetime(2026, 4, 29, 14, tzinfo=timezone.utc))

        self.assertEqual(len(bars), 2)
        self.assertEqual([b.timestamp.isoformat() for b in bars], ["2026-04-29T13:00:00+00:00", "2026-04-29T13:10:00+00:00"])
        self.assertEqual(adapter.requests[0][0], "/v2/stocks/SPY/bars")

    def test_latest_quote_normalises_valid_payload(self) -> None:
        adapter = FakeAlpacaAdapter(
            {
                "/v2/stocks/quotes/latest": {
                    "quotes": {
                        "SPY": {
                            "t": "2026-04-29T13:00:00Z",
                            "bp": "100.10",
                            "ap": "100.20",
                            "bs": "3",
                        }
                    }
                }
            }
        )

        quote = adapter.get_latest_quote("spy")

        self.assertEqual(quote.timestamp.isoformat(), "2026-04-29T13:00:00+00:00")
        self.assertEqual(quote.bid, 100.10)
        self.assertEqual(quote.ask, 100.20)
        self.assertAlmostEqual(quote.last, 100.15)
        self.assertEqual(quote.size, 3.0)

    def test_latest_quote_rejects_crossed_market(self) -> None:
        adapter = FakeAlpacaAdapter(
            {
                "/v2/stocks/quotes/latest": {
                    "quotes": {
                        "SPY": {
                            "t": "2026-04-29T13:00:00Z",
                            "bp": "100.30",
                            "ap": "100.20",
                        }
                    }
                }
            }
        )

        with self.assertRaisesRegex(RuntimeError, "crossed"):
            adapter.get_latest_quote("SPY")


if __name__ == "__main__":
    unittest.main()
