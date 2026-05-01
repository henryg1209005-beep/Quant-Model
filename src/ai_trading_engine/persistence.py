from __future__ import annotations

import json
import sqlite3
import csv
import io
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

from ai_trading_engine.engine import CycleResult
from ai_trading_engine.models import AccountState, ClosedTrade
from ai_trading_engine.serialization import account_to_record, decision_to_record, trade_to_record


@dataclass(frozen=True)
class RuntimeConfig:
    symbol: str
    symbols: tuple[str, ...]
    multi_symbol_enabled: bool
    multi_symbol_shadow_enabled: bool
    multi_symbol_active: bool
    multi_symbol_shadow_symbols: tuple[str, ...]
    cycle_seconds: int
    llm_provider: str
    data_provider: str
    execution_provider: str
    auto_retrain_enabled: bool
    auto_retrain_check_seconds: int
    auto_retrain_interval_hours: int
    auto_retrain_min_new_trades: int
    auto_retrain_lookback: int
    data_acceleration_mode: bool
    acceleration_active: bool
    entry_confluence_min: float
    ev_min_ticks: float
    max_spread_bps: float
    cost_model_enabled: bool
    cost_slippage_bps_per_side: float
    cost_fee_per_share: float
    cost_min_fee_per_order: float
    max_position_size: int
    max_trades_per_day: int
    max_daily_drawdown: float
    macro_context_enabled: bool
    macro_provider: str
    macro_snapshot_path: str
    economic_calendar_enabled: bool
    economic_calendar_provider: str
    economic_calendar_lookahead_days: int
    economic_calendar_high_impact_window_minutes: int
    economic_calendar_block_high_impact: bool
    finnhub_context_enabled: bool
    finnhub_news_lookback_days: int
    finnhub_news_max_headlines: int
    finnhub_news_min_score: float
    finnhub_earnings_lookahead_days: int
    earnings_risk_window_days: int
    model_monitoring_enabled: bool
    model_monitor_horizon_minutes: int
    model_monitor_short_window_days: int
    model_monitor_long_window_days: int
    model_monitor_min_labels: int
    model_monitor_breach_streak: int
    model_monitor_safe_mode_enabled: bool
    max_position_age_minutes: int
    broker_sync_stale_seconds: int
    stale_position_force_close: bool
    require_broker_protection_orders: bool
    protection_check_grace_seconds: int


class PersistenceLike(Protocol):
    def save_runtime_config(self, cfg: RuntimeConfig) -> None: ...
    def get_runtime_config(self) -> dict[str, Any] | None: ...
    def save_cycle(self, cycle: CycleResult, metadata: dict[str, Any] | None = None) -> None: ...
    def save_data_sample(self, cycle: CycleResult, metadata: dict[str, Any] | None = None) -> None: ...
    def list_data_samples(self, limit: int = 1000) -> list[dict[str, Any]]: ...
    def export_data_samples_csv(self, limit: int = 10000) -> str: ...
    def save_account(self, account: AccountState) -> None: ...
    def save_closed_trade(self, trade: ClosedTrade) -> None: ...
    def list_decisions(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def list_closed_trades(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def latest_account_snapshot(self) -> dict[str, Any] | None: ...


class _SqlitePersistence:
    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cycle_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    note TEXT NOT NULL,
                    llm_raw TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    dashboard TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS data_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    collection_role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    direction TEXT,
                    confidence REAL NOT NULL,
                    size INTEGER NOT NULL,
                    sl_ticks INTEGER NOT NULL,
                    tp_ticks INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    data_provider TEXT NOT NULL,
                    execution_provider TEXT NOT NULL,
                    llm_provider TEXT NOT NULL,
                    note TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_data_samples_ts ON data_samples(ts);
                CREATE INDEX IF NOT EXISTS idx_data_samples_symbol ON data_samples(symbol);
                """
            )

    @staticmethod
    def _sample_record(cycle: CycleResult, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        meta = dict(metadata or {})
        cycle_meta = dict(cycle.metadata or {})
        macro = dict(cycle_meta.get("macro") or {})
        economic_calendar = dict(cycle_meta.get("economic_calendar") or {})
        finnhub_context = dict(cycle_meta.get("finnhub_context") or {})
        quote = dict(cycle_meta.get("quote") or {})
        indicators = dict(cycle_meta.get("indicators") or {})
        features = dict(cycle_meta.get("features") or {})
        sample_quality = dict(cycle_meta.get("sample_quality") or {})
        quality_flags = dict(sample_quality.get("flags") or {})
        providers = dict(meta.get("providers") or {})
        thresholds = dict(meta.get("thresholds") or {})
        costs = dict(meta.get("costs") or {})
        decision = decision_to_record(cycle.decision)
        return {
            "timestamp": cycle.timestamp.isoformat(),
            "symbol": str(meta.get("symbol") or ""),
            "collection_role": str(meta.get("collection_role") or "primary"),
            "shadow_execution": bool(meta.get("shadow_execution", False)),
            "model_name": str(meta.get("model_name") or ""),
            "model_version": str(meta.get("model_version") or ""),
            "mode": str(meta.get("mode") or ""),
            "acceleration_active": bool(meta.get("acceleration_active", False)),
            "data_provider": str(providers.get("data") or ""),
            "execution_provider": str(providers.get("execution") or ""),
            "llm_provider": str(providers.get("llm") or ""),
            "action": str(decision.get("action") or ""),
            "direction": decision.get("direction"),
            "confidence": float(decision.get("confidence") or 0.0),
            "size": int(decision.get("size") or 0),
            "sl_ticks": int(decision.get("sl_ticks") or 0),
            "tp_ticks": int(decision.get("tp_ticks") or 0),
            "reasoning": str(decision.get("reasoning") or ""),
            "forecast_direction": decision.get("forecast_direction"),
            "forecast_confidence": float(decision.get("forecast_confidence") or 0.0),
            "forecast_horizon_minutes": int(decision.get("forecast_horizon_minutes") or 0),
            "note": cycle.note,
            "entry_confluence_min": float(thresholds.get("entry_confluence_min") or 0.0),
            "ev_min_ticks": float(thresholds.get("ev_min_ticks") or 0.0),
            "max_spread_bps": float(thresholds.get("max_spread_bps") or 0.0),
            "edge_min_win_rate": float(thresholds.get("edge_min_win_rate") or 0.0),
            "edge_min_expectancy": float(thresholds.get("edge_min_expectancy") or 0.0),
            "cost_model_enabled": bool(costs.get("enabled", False)),
            "macro_provider": str(macro.get("provider") or ""),
            "macro_risk_regime": str(macro.get("risk_regime") or ""),
            "macro_rate_regime": str(macro.get("rate_regime") or ""),
            "macro_inflation_regime": str(macro.get("inflation_regime") or ""),
            "macro_growth_regime": str(macro.get("growth_regime") or ""),
            "macro_policy_bias": str(macro.get("policy_bias") or ""),
            "macro_values": dict(macro.get("values") or {}),
            "macro_notes": str(macro.get("notes") or ""),
            "economic_calendar_provider": str(economic_calendar.get("provider") or ""),
            "economic_calendar_event_count": int(economic_calendar.get("event_count") or 0),
            "economic_calendar_high_impact_count": int(economic_calendar.get("high_impact_count") or 0),
            "economic_calendar_event_risk": str(economic_calendar.get("event_risk") or ""),
            "economic_calendar_next_event": dict(economic_calendar.get("next_event") or {}),
            "economic_calendar_events": list(economic_calendar.get("events") or []),
            "finnhub_news_risk": str(finnhub_context.get("news_risk") or ""),
            "finnhub_earnings_risk": str(finnhub_context.get("earnings_risk") or ""),
            "finnhub_news_sentiment_label": str(finnhub_context.get("news_sentiment_label") or ""),
            "finnhub_news_sentiment_score": float(finnhub_context.get("news_sentiment_score") or 0.0),
            "finnhub_market_headlines": list(finnhub_context.get("market_headlines") or []),
            "finnhub_company_headlines": list(finnhub_context.get("company_headlines") or []),
            "finnhub_earnings": list(finnhub_context.get("earnings") or []),
            "quote_timestamp": str(quote.get("timestamp") or ""),
            "quote_bid": float(quote.get("bid") or 0.0),
            "quote_ask": float(quote.get("ask") or 0.0),
            "quote_last": float(quote.get("last") or 0.0),
            "quote_size": float(quote.get("size") or 0.0),
            "quote_spread_bps": float(quote.get("spread_bps") or 0.0),
            "quote_age_seconds": float(quote.get("age_seconds") or 0.0),
            "indicator_regime": str(indicators.get("regime") or ""),
            "indicator_atr14": float(indicators.get("atr14") or 0.0),
            "indicator_rsi14": float(indicators.get("rsi14") or 0.0),
            "indicator_macd_hist": float(indicators.get("macd_hist") or 0.0),
            "feature_trend_score": float(features.get("trend_score") or 0.0),
            "feature_momentum_score": float(features.get("momentum_score") or 0.0),
            "feature_mean_reversion_score": float(features.get("mean_reversion_score") or 0.0),
            "feature_sr_distance_atr": float(features.get("sr_distance_atr") or 0.0),
            "feature_volatility_label": str(features.get("volatility_label") or ""),
            "feature_volume_label": str(features.get("volume_label") or ""),
            "feature_sr_label": str(features.get("sr_label") or ""),
            "feature_dimensions": dict(features.get("dimensions") or {}),
            "feature_key_levels": dict(features.get("key_levels") or {}),
            "sample_quality_good": bool(sample_quality.get("good", False)),
            "sample_quality_flags": quality_flags,
            "session_bucket": str(sample_quality.get("session_bucket") or ""),
            "session_meta": str(sample_quality.get("session_meta") or ""),
            "quality_outside_session": bool(quality_flags.get("outside_session", False)),
            "quality_quote_stale": bool(quality_flags.get("quote_stale", False)),
            "quality_spread_too_wide": bool(quality_flags.get("spread_too_wide", False)),
            "quality_macro_fallback": bool(quality_flags.get("macro_fallback", False)),
            "quality_economic_calendar_error": bool(quality_flags.get("economic_calendar_error", False)),
            "quality_finnhub_context_error": bool(quality_flags.get("finnhub_context_error", False)),
            "quality_missing_forecast": bool(quality_flags.get("missing_forecast", False)),
            "llm_raw": cycle.llm_raw,
            "dashboard": cycle.dashboard,
        }

    def save_runtime_config(self, cfg: RuntimeConfig) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        payload = json.dumps(cfg.__dict__, ensure_ascii=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runtime_config(id, payload_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (payload, now),
            )

    def get_runtime_config(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT payload_json, updated_at FROM runtime_config WHERE id = 1").fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload_json"])
            payload["updated_at"] = row["updated_at"]
            return payload

    def save_cycle(self, cycle: CycleResult, metadata: dict[str, Any] | None = None) -> None:
        decision_record = decision_to_record(cycle.decision)
        if metadata:
            decision_record["metadata"] = metadata
        if cycle.metadata:
            decision_record["cycle_metadata"] = cycle.metadata
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO cycle_logs(ts, note, llm_raw, decision_json, dashboard)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cycle.timestamp.isoformat(),
                    cycle.note,
                    cycle.llm_raw,
                    json.dumps(decision_record, ensure_ascii=True),
                    cycle.dashboard,
                ),
            )

    def save_data_sample(self, cycle: CycleResult, metadata: dict[str, Any] | None = None) -> None:
        sample = self._sample_record(cycle, metadata)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO data_samples(
                    ts, symbol, collection_role, action, direction, confidence, size,
                    sl_ticks, tp_ticks, mode, data_provider, execution_provider, llm_provider,
                    note, reasoning, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample["timestamp"],
                    sample["symbol"],
                    sample["collection_role"],
                    sample["action"],
                    sample["direction"],
                    sample["confidence"],
                    sample["size"],
                    sample["sl_ticks"],
                    sample["tp_ticks"],
                    sample["mode"],
                    sample["data_provider"],
                    sample["execution_provider"],
                    sample["llm_provider"],
                    sample["note"],
                    sample["reasoning"],
                    json.dumps(sample, ensure_ascii=True),
                ),
            )

    def list_data_samples(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM data_samples ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def export_data_samples_csv(self, limit: int = 10000) -> str:
        rows = self.list_data_samples(limit)
        fieldnames = [
            "timestamp",
            "symbol",
            "collection_role",
            "shadow_execution",
            "model_name",
            "model_version",
            "mode",
            "acceleration_active",
            "data_provider",
            "execution_provider",
            "llm_provider",
            "action",
            "direction",
            "confidence",
            "size",
            "sl_ticks",
            "tp_ticks",
            "reasoning",
            "forecast_direction",
            "forecast_confidence",
            "forecast_horizon_minutes",
            "note",
            "entry_confluence_min",
            "ev_min_ticks",
            "max_spread_bps",
            "edge_min_win_rate",
            "edge_min_expectancy",
            "cost_model_enabled",
            "macro_provider",
            "macro_risk_regime",
            "macro_rate_regime",
            "macro_inflation_regime",
            "macro_growth_regime",
            "macro_policy_bias",
            "macro_notes",
            "economic_calendar_provider",
            "economic_calendar_event_count",
            "economic_calendar_high_impact_count",
            "economic_calendar_event_risk",
            "finnhub_news_risk",
            "finnhub_earnings_risk",
            "finnhub_news_sentiment_label",
            "finnhub_news_sentiment_score",
            "quote_timestamp",
            "quote_bid",
            "quote_ask",
            "quote_last",
            "quote_size",
            "quote_spread_bps",
            "quote_age_seconds",
            "indicator_regime",
            "indicator_atr14",
            "indicator_rsi14",
            "indicator_macd_hist",
            "feature_trend_score",
            "feature_momentum_score",
            "feature_mean_reversion_score",
            "feature_sr_distance_atr",
            "feature_volatility_label",
            "feature_volume_label",
            "feature_sr_label",
            "sample_quality_good",
            "session_bucket",
            "quality_outside_session",
            "quality_quote_stale",
            "quality_spread_too_wide",
            "quality_macro_fallback",
            "quality_economic_calendar_error",
            "quality_finnhub_context_error",
            "quality_missing_forecast",
        ]
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()

    def save_account(self, account: AccountState) -> None:
        payload = json.dumps(account_to_record(account), ensure_ascii=True)
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("INSERT INTO account_snapshots(ts, payload_json) VALUES (?, ?)", (now, payload))

    def save_closed_trade(self, trade: ClosedTrade) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        payload = json.dumps(trade_to_record(trade), ensure_ascii=True)
        with self._conn() as conn:
            conn.execute("INSERT INTO closed_trades(ts, payload_json) VALUES (?, ?)", (now, payload))

    def list_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, note, decision_json FROM cycle_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"timestamp": row["ts"], "note": row["note"], "decision": json.loads(row["decision_json"])} for row in rows]

    def list_closed_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, payload_json FROM closed_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"timestamp": row["ts"], **json.loads(row["payload_json"])} for row in rows]

    def latest_account_snapshot(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT payload_json FROM account_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])


class _PostgresPersistence:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        try:
            import psycopg
            from psycopg.rows import dict_row
        except Exception as exc:
            raise RuntimeError("Postgres backend requires `psycopg[binary]` installed") from exc

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = self._psycopg.connect(self._dsn, row_factory=self._dict_row)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS account_snapshots (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cycle_logs (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        note TEXT NOT NULL,
                        llm_raw TEXT NOT NULL,
                        decision_json TEXT NOT NULL,
                        dashboard TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS closed_trades (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_config (
                        id INTEGER PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS data_samples (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        collection_role TEXT NOT NULL,
                        action TEXT NOT NULL,
                        direction TEXT,
                        confidence DOUBLE PRECISION NOT NULL,
                        size INTEGER NOT NULL,
                        sl_ticks INTEGER NOT NULL,
                        tp_ticks INTEGER NOT NULL,
                        mode TEXT NOT NULL,
                        data_provider TEXT NOT NULL,
                        execution_provider TEXT NOT NULL,
                        llm_provider TEXT NOT NULL,
                        note TEXT NOT NULL,
                        reasoning TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_data_samples_ts ON data_samples(ts);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_data_samples_symbol ON data_samples(symbol);")

    def save_runtime_config(self, cfg: RuntimeConfig) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        payload = json.dumps(cfg.__dict__, ensure_ascii=True)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runtime_config(id, payload_json, updated_at)
                    VALUES (1, %s, %s)
                    ON CONFLICT(id) DO UPDATE SET
                      payload_json=EXCLUDED.payload_json,
                      updated_at=EXCLUDED.updated_at
                    """,
                    (payload, now),
                )

    def get_runtime_config(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload_json, updated_at FROM runtime_config WHERE id = 1")
                row = cur.fetchone()
                if row is None:
                    return None
                payload = json.loads(row["payload_json"])
                payload["updated_at"] = row["updated_at"]
                return payload

    def save_cycle(self, cycle: CycleResult, metadata: dict[str, Any] | None = None) -> None:
        decision_record = decision_to_record(cycle.decision)
        if metadata:
            decision_record["metadata"] = metadata
        if cycle.metadata:
            decision_record["cycle_metadata"] = cycle.metadata
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cycle_logs(ts, note, llm_raw, decision_json, dashboard)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        cycle.timestamp.isoformat(),
                        cycle.note,
                        cycle.llm_raw,
                        json.dumps(decision_record, ensure_ascii=True),
                        cycle.dashboard,
                    ),
                )

    def save_data_sample(self, cycle: CycleResult, metadata: dict[str, Any] | None = None) -> None:
        sample = _SqlitePersistence._sample_record(cycle, metadata)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_samples(
                        ts, symbol, collection_role, action, direction, confidence, size,
                        sl_ticks, tp_ticks, mode, data_provider, execution_provider, llm_provider,
                        note, reasoning, payload_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sample["timestamp"],
                        sample["symbol"],
                        sample["collection_role"],
                        sample["action"],
                        sample["direction"],
                        sample["confidence"],
                        sample["size"],
                        sample["sl_ticks"],
                        sample["tp_ticks"],
                        sample["mode"],
                        sample["data_provider"],
                        sample["execution_provider"],
                        sample["llm_provider"],
                        sample["note"],
                        sample["reasoning"],
                        json.dumps(sample, ensure_ascii=True),
                    ),
                )

    def list_data_samples(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload_json FROM data_samples ORDER BY id DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def export_data_samples_csv(self, limit: int = 10000) -> str:
        rows = self.list_data_samples(limit)
        fieldnames = [
            "timestamp",
            "symbol",
            "collection_role",
            "shadow_execution",
            "model_name",
            "model_version",
            "mode",
            "acceleration_active",
            "data_provider",
            "execution_provider",
            "llm_provider",
            "action",
            "direction",
            "confidence",
            "size",
            "sl_ticks",
            "tp_ticks",
            "reasoning",
            "forecast_direction",
            "forecast_confidence",
            "forecast_horizon_minutes",
            "note",
            "entry_confluence_min",
            "ev_min_ticks",
            "max_spread_bps",
            "edge_min_win_rate",
            "edge_min_expectancy",
            "cost_model_enabled",
            "macro_provider",
            "macro_risk_regime",
            "macro_rate_regime",
            "macro_inflation_regime",
            "macro_growth_regime",
            "macro_policy_bias",
            "macro_notes",
            "economic_calendar_provider",
            "economic_calendar_event_count",
            "economic_calendar_high_impact_count",
            "economic_calendar_event_risk",
            "finnhub_news_risk",
            "finnhub_earnings_risk",
            "finnhub_news_sentiment_label",
            "finnhub_news_sentiment_score",
            "quote_timestamp",
            "quote_bid",
            "quote_ask",
            "quote_last",
            "quote_size",
            "quote_spread_bps",
            "quote_age_seconds",
            "indicator_regime",
            "indicator_atr14",
            "indicator_rsi14",
            "indicator_macd_hist",
            "feature_trend_score",
            "feature_momentum_score",
            "feature_mean_reversion_score",
            "feature_sr_distance_atr",
            "feature_volatility_label",
            "feature_volume_label",
            "feature_sr_label",
            "sample_quality_good",
            "session_bucket",
            "quality_outside_session",
            "quality_quote_stale",
            "quality_spread_too_wide",
            "quality_macro_fallback",
            "quality_economic_calendar_error",
            "quality_finnhub_context_error",
            "quality_missing_forecast",
        ]
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()

    def save_account(self, account: AccountState) -> None:
        payload = json.dumps(account_to_record(account), ensure_ascii=True)
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO account_snapshots(ts, payload_json) VALUES (%s, %s)", (now, payload))

    def save_closed_trade(self, trade: ClosedTrade) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        payload = json.dumps(trade_to_record(trade), ensure_ascii=True)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO closed_trades(ts, payload_json) VALUES (%s, %s)", (now, payload))

    def list_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ts, note, decision_json FROM cycle_logs ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        return [{"timestamp": row["ts"], "note": row["note"], "decision": json.loads(row["decision_json"])} for row in rows]

    def list_closed_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ts, payload_json FROM closed_trades ORDER BY id DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
        return [{"timestamp": row["ts"], **json.loads(row["payload_json"])} for row in rows]

    def latest_account_snapshot(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload_json FROM account_snapshots ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])


class Persistence:
    """Backend-agnostic persistence facade."""

    def __init__(self, db_path: str, database_url: str = "") -> None:
        if database_url:
            self._backend: PersistenceLike = _PostgresPersistence(database_url)
        else:
            self._backend = _SqlitePersistence(db_path)

    def __getattr__(self, name: str):
        return getattr(self._backend, name)
