from __future__ import annotations

import unittest

from ai_trading_engine.adapters.mock_adapter import MockMarketDataAdapter
from ai_trading_engine.bar_builder import BarBuilder


class TestBarBuilder(unittest.TestCase):
    def test_bar_builder_shapes(self) -> None:
        adapter = MockMarketDataAdapter(seed=1)
        builder = BarBuilder(adapter)
        tf = builder.build_multi_timeframe("ES", 5, 1, 60, 15, 15, 12)

        self.assertEqual(len(tf.primary), 15)
        self.assertEqual(len(tf.short), 15)
        self.assertEqual(len(tf.long), 12)

        ordered = all(tf.primary[i].timestamp < tf.primary[i + 1].timestamp for i in range(len(tf.primary) - 1))
        self.assertTrue(ordered)


if __name__ == "__main__":
    unittest.main()
