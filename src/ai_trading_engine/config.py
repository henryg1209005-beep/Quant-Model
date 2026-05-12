from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader to avoid external dependency for local runs."""
    cwd_env = Path.cwd() / ".env"
    project_env = Path(__file__).resolve().parents[2] / ".env"
    env_path = cwd_env if cwd_env.exists() else project_env
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # Prefer project .env values for deterministic local research runs.
            os.environ[key] = value


_load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _get_env(name: str, default: str = "", aliases: tuple[str, ...] = ()) -> str:
    raw = os.getenv(name)
    if raw is not None and raw.strip() != "":
        return raw
    for alias in aliases:
        alt = os.getenv(alias)
        if alt is not None and alt.strip() != "":
            return alt
    return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    values: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        item = part.strip().upper()
        if not item or item in seen:
            continue
        seen.add(item)
        values.append(item)
    return tuple(values)


def _get_csv_lower(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    values: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        item = part.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        values.append(item)
    return tuple(values)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_runtime_path(raw_path: str) -> str:
    raw = str(raw_path or "").strip()
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    if normalized == "src/data" or normalized.startswith("src/data/"):
        suffix = normalized[len("src/data"):].lstrip("/")
        mapped = Path("data") / suffix if suffix else Path("data")
        return str(mapped).replace("\\", "/")
    return raw


def _migrate_src_data_to_data() -> None:
    src_root = _project_root() / "src" / "data"
    dst_root = _project_root() / "data"
    if not src_root.exists() or not src_root.is_dir():
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    for source in src_root.rglob("*"):
        if source.is_dir():
            continue
        rel = source.relative_to(src_root)
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)


@dataclass(frozen=True)
class Settings:
    env: str
    symbol: str
    symbols: tuple[str, ...]
    multi_symbol_enabled: bool
    multi_symbol_shadow_enabled: bool
    multi_symbol_paper_only: bool
    timezone: str
    primary_timeframe_min: int
    short_timeframe_min: int
    long_timeframe_min: int
    primary_bars: int
    short_bars: int
    long_bars: int
    cycle_seconds: int
    max_position_size: int
    min_sl_ticks: int
    min_tp_ticks: int
    max_trades_per_day: int
    max_daily_drawdown: float
    llm_provider: str
    llm_two_tier_enabled: bool
    llm_two_tier_secondary_provider: str
    llm_two_tier_escalate_on_trade: bool
    llm_two_tier_require_confidence_band: bool
    llm_two_tier_min_confidence: float
    llm_two_tier_max_confidence: float
    llm_max_context_chars: int
    llm_context_recent_bars: int
    llm_max_output_tokens: int
    llm_state_change_gate_enabled: bool
    llm_state_change_min_seconds: int
    llm_state_change_price_bps: float
    llm_state_change_trend_delta: float
    llm_state_change_momentum_delta: float
    llm_request_retries: int
    llm_retry_backoff_ms: int
    llm_provider_health_window: int
    llm_provider_unhealthy_fail_rate: float
    llm_provider_cooldown_seconds: int
    openai_model: str
    openai_reasoning_effort: str
    openai_temperature: float | None
    gemini_model: str
    gemini_secondary_model: str
    gemini_temperature: float | None
    gemini_timeout_seconds: int
    data_provider: str
    execution_provider: str
    alpaca_key_id: str
    alpaca_secret_key: str
    alpaca_data_url: str
    alpaca_trading_url: str
    enable_session_filter: bool
    session_start_et: str
    session_end_et: str
    max_spread_bps: float
    entry_confluence_min: float
    price_tick_size: float
    risk_per_trade_usd: float
    ev_min_ticks: float
    trade_blocked_symbols: tuple[str, ...]
    trading_kill_switch_enabled: bool
    trading_kill_switch_reason: str
    signed_bps_kill_enabled: bool
    signed_bps_kill_threshold: float
    signed_bps_kill_min_labels: int
    signed_bps_kill_horizon_minutes: int
    signed_bps_kill_refresh_seconds: int
    symbol_gate_enabled: bool
    symbol_gate_horizon_minutes: int
    symbol_gate_min_labels: int
    symbol_gate_min_signed_bps: float
    symbol_gate_min_accuracy: float
    symbol_gate_refresh_seconds: int
    regime_gate_enabled: bool
    regime_gate_horizon_minutes: int
    regime_gate_refresh_seconds: int
    regime_gate_min_labels: int
    regime_gate_min_signed_bps: float
    regime_gate_min_accuracy: float
    best_horizon_gate_enabled: bool
    best_horizon_gate_min_labels: int
    best_horizon_gate_refresh_seconds: int
    best_horizon_choices: str
    no_trade_windows_et: str
    sample_balance_enabled: bool
    sample_balance_max_share: float
    sample_balance_min_per_symbol_per_day: int
    coverage_guard_enabled: bool
    coverage_guard_daily_label_target: int
    coverage_guard_relax_label_factor: float
    confidence_control_enabled: bool
    confidence_control_horizon_minutes: int
    confidence_control_refresh_seconds: int
    confidence_control_min_bin_count: int
    confidence_control_min_symbol_labels: int
    confidence_control_min_threshold_labels: int
    confidence_control_default_min_confidence: float
    confidence_calibration_mode: str
    confidence_calibration_min_populated_bins: int
    confidence_calibration_min_labels_per_bin: int
    confidence_calibration_allowed_sources: tuple[str, ...]
    calibration_gate_enabled: bool
    calibration_gate_stress_bps: float
    data_acceleration_mode: bool
    accel_relax_confluence: float
    accel_relax_ev_ticks: float
    accel_spread_mult: float
    accel_session_start_et: str
    accel_session_end_et: str
    accel_edge_min_win_rate_delta: float
    accel_edge_min_expectancy_delta: float
    edge_window_trades: int
    edge_min_win_rate: float
    edge_min_expectancy: float
    edge_throttle_size_mult: float
    edge_monitor_state_path: str
    shadow_challenger_enabled: bool
    shadow_confluence_bump: float
    shadow_ev_bump: float
    macro_context_enabled: bool
    macro_provider: str
    macro_snapshot_path: str
    fred_api_key: str
    macro_request_timeout_seconds: int
    macro_risk_regime: str
    macro_rate_regime: str
    macro_inflation_regime: str
    macro_growth_regime: str
    macro_policy_bias: str
    macro_notes: str
    economic_calendar_enabled: bool
    economic_calendar_provider: str
    finnhub_api_key: str
    economic_calendar_lookahead_days: int
    economic_calendar_request_timeout_seconds: int
    economic_calendar_high_impact_window_minutes: int
    economic_calendar_block_high_impact: bool
    finnhub_context_enabled: bool
    finnhub_news_lookback_days: int
    finnhub_news_max_headlines: int
    finnhub_news_min_score: float
    finnhub_earnings_lookahead_days: int
    finnhub_request_timeout_seconds: int
    earnings_risk_window_days: int
    adaptive_enabled: bool
    adaptive_state_path: str
    autonomous_live_enabled: bool
    go_live_min_closed_trades: int
    go_live_min_win_rate: float
    go_live_min_profit_factor: float
    go_live_min_expectancy: float
    go_live_max_drawdown_pct: float
    go_live_max_hold_rate: float
    go_live_metrics_lookback: int
    auto_start_worker: bool
    auto_retrain_enabled: bool
    auto_retrain_check_seconds: int
    auto_retrain_interval_hours: int
    auto_retrain_min_new_trades: int
    auto_retrain_neg_expectancy_lookback: int
    auto_retrain_lookback: int
    auto_research_enabled: bool
    auto_research_lookback: int
    auto_research_folds: int
    auto_research_min_train: int
    auto_research_min_test: int
    auto_research_bins: int
    auto_retrain_state_path: str
    promotion_state_path: str
    promotion_min_folds: int
    promotion_min_model_selected_trades: int
    promotion_min_expectancy: float
    promotion_min_net_pnl_edge: float
    promotion_require_recommendation_promote: bool
    pretrade_max_order_notional_usd: float
    pretrade_max_total_exposure_usd: float
    pretrade_max_quote_age_seconds: int
    pretrade_max_spread_bps: float
    pretrade_min_buying_power_usd: float
    pretrade_duplicate_cooldown_seconds: int
    sample_max_quote_age_seconds: int
    research_max_quote_age_seconds: float
    research_allowed_session_buckets: str
    promotion_cell_min_labels: int
    promotion_cost_stress_multiplier: float
    promotion_require_ci95_positive: bool
    promotion_require_oos_positive: bool
    research_freeze_enabled: bool
    weekly_experiments_enabled: bool
    weekly_experiments_max_per_week: int
    weekly_experiment_min_labels: int
    weekly_experiment_stress_multiplier: float
    auto_promotion_gate_enabled: bool
    auto_promotion_min_labels: int
    auto_promotion_min_stressed_signed_bps: float
    auto_promotion_require_oos_positive: bool
    model_monitoring_enabled: bool
    model_monitor_lookback: int
    model_monitor_horizon_minutes: int
    model_monitor_min_confidence: float
    model_monitor_short_window_days: int
    model_monitor_long_window_days: int
    model_monitor_min_labels: int
    model_monitor_min_accuracy_delta: float
    model_monitor_min_signed_bps_delta: float
    model_monitor_max_brier_delta: float
    model_monitor_breach_streak: int
    model_monitor_safe_mode_enabled: bool
    model_monitor_state_path: str
    max_position_age_minutes: int
    broker_sync_stale_seconds: int
    stale_position_force_close: bool
    require_broker_protection_orders: bool
    protection_check_grace_seconds: int
    cost_model_enabled: bool
    cost_slippage_bps_per_side: float
    cost_fee_per_share: float
    cost_min_fee_per_order: float
    cost_apply_in_paper: bool
    cost_apply_in_mock: bool
    automation_quality_checks_enabled: bool
    automation_quality_check_interval_seconds: int
    automation_quality_window_minutes: int
    automation_quality_min_samples: int
    automation_quality_min_samples_offsession_factor: float
    automation_quality_min_samples_floor: int
    automation_quality_enforce_volume_offsession: bool
    automation_quality_max_stale_quote_ratio: float
    automation_quality_max_missing_forecast_ratio: float
    automation_quality_max_bad_sample_ratio: float
    automation_quality_throttle_cycle_seconds: int
    sample_flow_guard_enabled: bool
    sample_flow_guard_stall_minutes_in_session: int
    automation_cost_guard_enabled: bool
    automation_cost_daily_budget_usd: float
    automation_cost_warn_fraction: float
    automation_cost_hard_block: bool
    automation_cost_est_primary_call_usd: float
    automation_cost_est_secondary_call_usd: float
    automation_cost_mode_enabled: bool
    automation_cost_mode_cycle_seconds: int
    automation_cost_mode_skip_low_volume_collect: bool
    automation_cost_mode_disable_finnhub_context: bool
    automation_cost_mode_disable_economic_calendar: bool
    budget_lock_enabled: bool
    budget_lock_in_session_only: bool
    budget_lock_max_llm_calls_per_hour: int
    budget_lock_state_change_min_seconds: int
    budget_lock_disable_secondary_escalation: bool
    automation_auto_recovery_enabled: bool
    automation_auto_recovery_cooldown_seconds: int
    robust_allowlist_enabled: bool
    robust_allowlist_horizon_minutes: int
    robust_allowlist_min_labels: int
    robust_allowlist_n_bootstrap: int
    robust_allowlist_cost_stress_multiplier: float
    robust_allowlist_require_oos_positive: bool
    robust_policy_tiering_enabled: bool
    robust_policy_tier_coverage_hours_strict: float
    robust_policy_tier_coverage_hours_balanced: float
    robust_policy_tier_strict_min_labels: int
    robust_policy_tier_balanced_min_labels: int
    robust_policy_tier_explore_min_labels: int
    robust_policy_tier_strict_cost_stress_multiplier: float
    robust_policy_tier_balanced_cost_stress_multiplier: float
    robust_policy_tier_explore_cost_stress_multiplier: float
    cell_cooldown_enabled: bool
    cell_cooldown_minutes: int
    cell_cooldown_min_labels: int
    cell_cooldown_signed_bps_kill: float
    symbol_quarantine_enabled: bool
    symbol_quarantine_horizon_minutes: int
    symbol_quarantine_min_labels: int
    symbol_quarantine_signed_bps_kill: float
    symbol_quarantine_minutes: int
    symbol_quarantine_recover_min_labels: int
    symbol_quarantine_recover_signed_bps: float
    autonomous_research_enabled: bool
    autonomous_research_check_seconds: int
    autonomous_research_suite_interval_minutes: int
    autonomous_self_scan_enabled: bool
    autonomous_self_scan_interval_minutes: int
    autonomous_self_scan_lookback: int
    audit_log_path: str
    app_db_path: str
    database_url: str
    app_auth_enabled: bool
    app_auth_username: str
    app_auth_password: str
    notifications_enabled: bool
    notifications_path: str
    api_host: str
    api_port: int


DEFAULT_SETTINGS = Settings(
    env=os.getenv("ENV", "dev"),
    symbol=os.getenv("SYMBOL", "SPY"),
    symbols=_get_csv("SYMBOLS", f"{os.getenv('SYMBOL', 'SPY')},QQQ,IWM"),
    multi_symbol_enabled=_get_bool("MULTI_SYMBOL_ENABLED", True),
    multi_symbol_shadow_enabled=_get_bool("MULTI_SYMBOL_SHADOW_ENABLED", True),
    multi_symbol_paper_only=_get_bool("MULTI_SYMBOL_PAPER_ONLY", True),
    timezone=os.getenv("TIMEZONE", "America/New_York"),
    primary_timeframe_min=_get_int("PRIMARY_TIMEFRAME_MIN", 5),
    short_timeframe_min=_get_int("SHORT_TIMEFRAME_MIN", 1),
    long_timeframe_min=_get_int("LONG_TIMEFRAME_MIN", 60),
    primary_bars=_get_int("PRIMARY_BARS", 15),
    short_bars=_get_int("SHORT_BARS", 15),
    long_bars=_get_int("LONG_BARS", 12),
    cycle_seconds=_get_int("CYCLE_SECONDS", 30),
    max_position_size=_get_int("MAX_POSITION_SIZE", 10),
    min_sl_ticks=_get_int("MIN_SL_TICKS", 6),
    min_tp_ticks=_get_int("MIN_TP_TICKS", 8),
    max_trades_per_day=_get_int("MAX_TRADES_PER_DAY", 12),
    max_daily_drawdown=_get_float("MAX_DAILY_DRAWDOWN", 1500.0),
    llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
    llm_two_tier_enabled=_get_bool("LLM_TWO_TIER_ENABLED", True),
    llm_two_tier_secondary_provider=os.getenv("LLM_TWO_TIER_SECONDARY_PROVIDER", "gemini").strip().lower(),
    llm_two_tier_escalate_on_trade=_get_bool("LLM_TWO_TIER_ESCALATE_ON_TRADE", True),
    llm_two_tier_require_confidence_band=_get_bool("LLM_TWO_TIER_REQUIRE_CONFIDENCE_BAND", True),
    llm_two_tier_min_confidence=_get_float("LLM_TWO_TIER_MIN_CONFIDENCE", 0.52),
    llm_two_tier_max_confidence=_get_float("LLM_TWO_TIER_MAX_CONFIDENCE", 0.70),
    llm_max_context_chars=_get_int("LLM_MAX_CONTEXT_CHARS", 8000),
    llm_context_recent_bars=_get_int("LLM_CONTEXT_RECENT_BARS", 5),
    llm_max_output_tokens=_get_int("LLM_MAX_OUTPUT_TOKENS", 240),
    llm_state_change_gate_enabled=_get_bool("LLM_STATE_CHANGE_GATE_ENABLED", True),
    llm_state_change_min_seconds=_get_int("LLM_STATE_CHANGE_MIN_SECONDS", 60),
    llm_state_change_price_bps=_get_float("LLM_STATE_CHANGE_PRICE_BPS", 6.0),
    llm_state_change_trend_delta=_get_float("LLM_STATE_CHANGE_TREND_DELTA", 8.0),
    llm_state_change_momentum_delta=_get_float("LLM_STATE_CHANGE_MOMENTUM_DELTA", 8.0),
    llm_request_retries=_get_int("LLM_REQUEST_RETRIES", 2),
    llm_retry_backoff_ms=_get_int("LLM_RETRY_BACKOFF_MS", 250),
    llm_provider_health_window=_get_int("LLM_PROVIDER_HEALTH_WINDOW", 60),
    llm_provider_unhealthy_fail_rate=_get_float("LLM_PROVIDER_UNHEALTHY_FAIL_RATE", 0.25),
    llm_provider_cooldown_seconds=_get_int("LLM_PROVIDER_COOLDOWN_SECONDS", 300),
    openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
    openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower(),
    openai_temperature=_get_optional_float("OPENAI_TEMPERATURE"),
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    gemini_secondary_model=os.getenv("GEMINI_SECONDARY_MODEL", "gemini-2.5-pro"),
    gemini_temperature=_get_optional_float("GEMINI_TEMPERATURE"),
    gemini_timeout_seconds=_get_int("GEMINI_TIMEOUT_SECONDS", 30),
    data_provider=os.getenv("DATA_PROVIDER", "mock").strip().lower(),
    execution_provider=os.getenv("EXECUTION_PROVIDER", "mock").strip().lower(),
    alpaca_key_id=_get_env("ALPACA_KEY_ID", "", aliases=("ALPACA_API_KEY",)),
    alpaca_secret_key=_get_env("ALPACA_SECRET_KEY", "", aliases=("ALPACA_API_SECRET",)),
    alpaca_data_url=os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"),
    alpaca_trading_url=os.getenv("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets"),
    enable_session_filter=_get_bool("ENABLE_SESSION_FILTER", True),
    session_start_et=os.getenv("SESSION_START_ET", "09:35"),
    session_end_et=os.getenv("SESSION_END_ET", "15:45"),
    max_spread_bps=_get_float("MAX_SPREAD_BPS", 8.0),
    entry_confluence_min=_get_float("ENTRY_CONFLUENCE_MIN", 58.0),
    price_tick_size=_get_float("PRICE_TICK_SIZE", 0.01),
    risk_per_trade_usd=_get_float("RISK_PER_TRADE_USD", 75.0),
    ev_min_ticks=_get_float("EV_MIN_TICKS", 0.2),
    trade_blocked_symbols=_get_csv("TRADE_BLOCKED_SYMBOLS", ""),
    trading_kill_switch_enabled=_get_bool("TRADING_KILL_SWITCH_ENABLED", False),
    trading_kill_switch_reason=os.getenv("TRADING_KILL_SWITCH_REASON", "manual_kill_switch"),
    signed_bps_kill_enabled=_get_bool("SIGNED_BPS_KILL_ENABLED", True),
    signed_bps_kill_threshold=_get_float("SIGNED_BPS_KILL_THRESHOLD", -5.0),
    signed_bps_kill_min_labels=_get_int("SIGNED_BPS_KILL_MIN_LABELS", 100),
    signed_bps_kill_horizon_minutes=_get_int("SIGNED_BPS_KILL_HORIZON_MINUTES", 15),
    signed_bps_kill_refresh_seconds=_get_int("SIGNED_BPS_KILL_REFRESH_SECONDS", 300),
    symbol_gate_enabled=_get_bool("SYMBOL_GATE_ENABLED", True),
    symbol_gate_horizon_minutes=_get_int("SYMBOL_GATE_HORIZON_MINUTES", 15),
    symbol_gate_min_labels=_get_int("SYMBOL_GATE_MIN_LABELS", 50),
    symbol_gate_min_signed_bps=_get_float("SYMBOL_GATE_MIN_SIGNED_BPS", 0.0),
    symbol_gate_min_accuracy=_get_float("SYMBOL_GATE_MIN_ACCURACY", 0.5),
    symbol_gate_refresh_seconds=_get_int("SYMBOL_GATE_REFRESH_SECONDS", 300),
    regime_gate_enabled=_get_bool("REGIME_GATE_ENABLED", True),
    regime_gate_horizon_minutes=_get_int("REGIME_GATE_HORIZON_MINUTES", 15),
    regime_gate_refresh_seconds=_get_int("REGIME_GATE_REFRESH_SECONDS", 300),
    regime_gate_min_labels=_get_int("REGIME_GATE_MIN_LABELS", 35),
    regime_gate_min_signed_bps=_get_float("REGIME_GATE_MIN_SIGNED_BPS", 0.0),
    regime_gate_min_accuracy=_get_float("REGIME_GATE_MIN_ACCURACY", 0.5),
    best_horizon_gate_enabled=_get_bool("BEST_HORIZON_GATE_ENABLED", True),
    best_horizon_gate_min_labels=_get_int("BEST_HORIZON_GATE_MIN_LABELS", 35),
    best_horizon_gate_refresh_seconds=_get_int("BEST_HORIZON_GATE_REFRESH_SECONDS", 300),
    best_horizon_choices=os.getenv("BEST_HORIZON_CHOICES", "5,15,30"),
    no_trade_windows_et=os.getenv("NO_TRADE_WINDOWS_ET", "09:30-09:40,15:50-16:00"),
    sample_balance_enabled=_get_bool("SAMPLE_BALANCE_ENABLED", True),
    sample_balance_max_share=_get_float("SAMPLE_BALANCE_MAX_SHARE", 0.45),
    sample_balance_min_per_symbol_per_day=_get_int("SAMPLE_BALANCE_MIN_PER_SYMBOL_PER_DAY", 25),
    coverage_guard_enabled=_get_bool("COVERAGE_GUARD_ENABLED", True),
    coverage_guard_daily_label_target=_get_int("COVERAGE_GUARD_DAILY_LABEL_TARGET", 80),
    coverage_guard_relax_label_factor=_get_float("COVERAGE_GUARD_RELAX_LABEL_FACTOR", 0.6),
    confidence_control_enabled=_get_bool("CONFIDENCE_CONTROL_ENABLED", True),
    confidence_control_horizon_minutes=_get_int("CONFIDENCE_CONTROL_HORIZON_MINUTES", 15),
    confidence_control_refresh_seconds=_get_int("CONFIDENCE_CONTROL_REFRESH_SECONDS", 300),
    confidence_control_min_bin_count=_get_int("CONFIDENCE_CONTROL_MIN_BIN_COUNT", 20),
    confidence_control_min_symbol_labels=_get_int("CONFIDENCE_CONTROL_MIN_SYMBOL_LABELS", 40),
    confidence_control_min_threshold_labels=_get_int("CONFIDENCE_CONTROL_MIN_THRESHOLD_LABELS", 30),
    confidence_control_default_min_confidence=_get_float("CONFIDENCE_CONTROL_DEFAULT_MIN_CONFIDENCE", 0.55),
    confidence_calibration_mode=os.getenv("CONFIDENCE_CALIBRATION_MODE", "auto").strip().lower(),
    confidence_calibration_min_populated_bins=_get_int("CONFIDENCE_CALIBRATION_MIN_POPULATED_BINS", 3),
    confidence_calibration_min_labels_per_bin=_get_int("CONFIDENCE_CALIBRATION_MIN_LABELS_PER_BIN", 30),
    confidence_calibration_allowed_sources=_get_csv_lower(
        "CONFIDENCE_CALIBRATION_ALLOWED_SOURCES",
        "model,cached",
    ),
    calibration_gate_enabled=_get_bool("CALIBRATION_GATE_ENABLED", True),
    calibration_gate_stress_bps=_get_float("CALIBRATION_GATE_STRESS_BPS", 3.0),
    data_acceleration_mode=_get_bool("DATA_ACCELERATION_MODE", False),
    accel_relax_confluence=_get_float("ACCEL_RELAX_CONFLUENCE", 8.0),
    accel_relax_ev_ticks=_get_float("ACCEL_RELAX_EV_TICKS", 0.10),
    accel_spread_mult=_get_float("ACCEL_SPREAD_MULT", 1.5),
    accel_session_start_et=os.getenv("ACCEL_SESSION_START_ET", "09:35"),
    accel_session_end_et=os.getenv("ACCEL_SESSION_END_ET", "15:55"),
    accel_edge_min_win_rate_delta=_get_float("ACCEL_EDGE_MIN_WIN_RATE_DELTA", 0.08),
    accel_edge_min_expectancy_delta=_get_float("ACCEL_EDGE_MIN_EXPECTANCY_DELTA", 20.0),
    edge_window_trades=_get_int("EDGE_WINDOW_TRADES", 40),
    edge_min_win_rate=_get_float("EDGE_MIN_WIN_RATE", 0.42),
    edge_min_expectancy=_get_float("EDGE_MIN_EXPECTANCY", 0.0),
    edge_throttle_size_mult=_get_float("EDGE_THROTTLE_SIZE_MULT", 0.6),
    edge_monitor_state_path=os.getenv("EDGE_MONITOR_STATE_PATH", "data/edge_monitor_state.json"),
    shadow_challenger_enabled=_get_bool("SHADOW_CHALLENGER_ENABLED", True),
    shadow_confluence_bump=_get_float("SHADOW_CONFLUENCE_BUMP", 6.0),
    shadow_ev_bump=_get_float("SHADOW_EV_BUMP", 0.4),
    macro_context_enabled=_get_bool("MACRO_CONTEXT_ENABLED", True),
    macro_provider=os.getenv("MACRO_PROVIDER", "manual").strip().lower(),
    macro_snapshot_path=os.getenv("MACRO_SNAPSHOT_PATH", "data/macro_snapshot.json"),
    fred_api_key=os.getenv("FRED_API_KEY", ""),
    macro_request_timeout_seconds=_get_int("MACRO_REQUEST_TIMEOUT_SECONDS", 8),
    macro_risk_regime=os.getenv("MACRO_RISK_REGIME", "unknown"),
    macro_rate_regime=os.getenv("MACRO_RATE_REGIME", "unknown"),
    macro_inflation_regime=os.getenv("MACRO_INFLATION_REGIME", "unknown"),
    macro_growth_regime=os.getenv("MACRO_GROWTH_REGIME", "unknown"),
    macro_policy_bias=os.getenv("MACRO_POLICY_BIAS", "unknown"),
    macro_notes=os.getenv("MACRO_NOTES", ""),
    economic_calendar_enabled=_get_bool("ECONOMIC_CALENDAR_ENABLED", False),
    economic_calendar_provider=os.getenv("ECONOMIC_CALENDAR_PROVIDER", "finnhub").strip().lower(),
    finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
    economic_calendar_lookahead_days=_get_int("ECONOMIC_CALENDAR_LOOKAHEAD_DAYS", 7),
    economic_calendar_request_timeout_seconds=_get_int("ECONOMIC_CALENDAR_REQUEST_TIMEOUT_SECONDS", 8),
    economic_calendar_high_impact_window_minutes=_get_int("ECONOMIC_CALENDAR_HIGH_IMPACT_WINDOW_MINUTES", 90),
    economic_calendar_block_high_impact=_get_bool("ECONOMIC_CALENDAR_BLOCK_HIGH_IMPACT", False),
    finnhub_context_enabled=_get_bool("FINNHUB_CONTEXT_ENABLED", False),
    finnhub_news_lookback_days=_get_int("FINNHUB_NEWS_LOOKBACK_DAYS", 2),
    finnhub_news_max_headlines=_get_int("FINNHUB_NEWS_MAX_HEADLINES", 5),
    finnhub_news_min_score=_get_float("FINNHUB_NEWS_MIN_SCORE", 1.0),
    finnhub_earnings_lookahead_days=_get_int("FINNHUB_EARNINGS_LOOKAHEAD_DAYS", 14),
    finnhub_request_timeout_seconds=_get_int("FINNHUB_REQUEST_TIMEOUT_SECONDS", 8),
    earnings_risk_window_days=_get_int("EARNINGS_RISK_WINDOW_DAYS", 2),
    adaptive_enabled=_get_bool("ADAPTIVE_ENABLED", True),
    adaptive_state_path=os.getenv("ADAPTIVE_STATE_PATH", "data/adaptive_state.json"),
    autonomous_live_enabled=_get_bool("AUTONOMOUS_LIVE_ENABLED", False),
    go_live_min_closed_trades=_get_int("GO_LIVE_MIN_CLOSED_TRADES", 100),
    go_live_min_win_rate=_get_float("GO_LIVE_MIN_WIN_RATE", 0.52),
    go_live_min_profit_factor=_get_float("GO_LIVE_MIN_PROFIT_FACTOR", 1.20),
    go_live_min_expectancy=_get_float("GO_LIVE_MIN_EXPECTANCY", 0.0),
    go_live_max_drawdown_pct=_get_float("GO_LIVE_MAX_DRAWDOWN_PCT", 0.05),
    go_live_max_hold_rate=_get_float("GO_LIVE_MAX_HOLD_RATE", 0.90),
    go_live_metrics_lookback=_get_int("GO_LIVE_METRICS_LOOKBACK", 1000),
    auto_start_worker=_get_bool("AUTO_START_WORKER", False),
    auto_retrain_enabled=_get_bool("AUTO_RETRAIN_ENABLED", False),
    auto_retrain_check_seconds=_get_int("AUTO_RETRAIN_CHECK_SECONDS", 900),
    auto_retrain_interval_hours=_get_int("AUTO_RETRAIN_INTERVAL_HOURS", 168),
    auto_retrain_min_new_trades=_get_int("AUTO_RETRAIN_MIN_NEW_TRADES", 50),
    auto_retrain_neg_expectancy_lookback=_get_int("AUTO_RETRAIN_NEG_EXPECTANCY_LOOKBACK", 30),
    auto_retrain_lookback=_get_int("AUTO_RETRAIN_LOOKBACK", 5000),
    auto_research_enabled=_get_bool("AUTO_RESEARCH_ENABLED", True),
    auto_research_lookback=_get_int("AUTO_RESEARCH_LOOKBACK", 10000),
    auto_research_folds=_get_int("AUTO_RESEARCH_FOLDS", 4),
    auto_research_min_train=_get_int("AUTO_RESEARCH_MIN_TRAIN", 40),
    auto_research_min_test=_get_int("AUTO_RESEARCH_MIN_TEST", 20),
    auto_research_bins=_get_int("AUTO_RESEARCH_BINS", 10),
    auto_retrain_state_path=os.getenv("AUTO_RETRAIN_STATE_PATH", "data/automation_state.json"),
    promotion_state_path=os.getenv("PROMOTION_STATE_PATH", "data/promotion_state.json"),
    promotion_min_folds=_get_int("PROMOTION_MIN_FOLDS", 3),
    promotion_min_model_selected_trades=_get_int("PROMOTION_MIN_MODEL_SELECTED_TRADES", 40),
    promotion_min_expectancy=_get_float("PROMOTION_MIN_EXPECTANCY", 0.0),
    promotion_min_net_pnl_edge=_get_float("PROMOTION_MIN_NET_PNL_EDGE", 0.0),
    promotion_require_recommendation_promote=_get_bool("PROMOTION_REQUIRE_RECOMMENDATION_PROMOTE", True),
    pretrade_max_order_notional_usd=_get_float("PRETRADE_MAX_ORDER_NOTIONAL_USD", 25000.0),
    pretrade_max_total_exposure_usd=_get_float("PRETRADE_MAX_TOTAL_EXPOSURE_USD", 75000.0),
    pretrade_max_quote_age_seconds=_get_int("PRETRADE_MAX_QUOTE_AGE_SECONDS", 5),
    pretrade_max_spread_bps=_get_float("PRETRADE_MAX_SPREAD_BPS", 10.0),
    pretrade_min_buying_power_usd=_get_float("PRETRADE_MIN_BUYING_POWER_USD", 1000.0),
    pretrade_duplicate_cooldown_seconds=_get_int("PRETRADE_DUPLICATE_COOLDOWN_SECONDS", 15),
    sample_max_quote_age_seconds=_get_int("SAMPLE_MAX_QUOTE_AGE_SECONDS", 3600),
    research_max_quote_age_seconds=_get_float("RESEARCH_MAX_QUOTE_AGE_SECONDS", 5.0),
    research_allowed_session_buckets=os.getenv("RESEARCH_ALLOWED_SESSION_BUCKETS", "open,midday,close,unknown"),
    promotion_cell_min_labels=_get_int("PROMOTION_CELL_MIN_LABELS", 30),
    promotion_cost_stress_multiplier=_get_float("PROMOTION_COST_STRESS_MULTIPLIER", 1.5),
    promotion_require_ci95_positive=_get_bool("PROMOTION_REQUIRE_CI95_POSITIVE", True),
    promotion_require_oos_positive=_get_bool("PROMOTION_REQUIRE_OOS_POSITIVE", True),
    research_freeze_enabled=_get_bool("RESEARCH_FREEZE_ENABLED", True),
    weekly_experiments_enabled=_get_bool("WEEKLY_EXPERIMENTS_ENABLED", True),
    weekly_experiments_max_per_week=_get_int("WEEKLY_EXPERIMENTS_MAX_PER_WEEK", 3),
    weekly_experiment_min_labels=_get_int("WEEKLY_EXPERIMENT_MIN_LABELS", 120),
    weekly_experiment_stress_multiplier=_get_float("WEEKLY_EXPERIMENT_STRESS_MULTIPLIER", 1.5),
    auto_promotion_gate_enabled=_get_bool("AUTO_PROMOTION_GATE_ENABLED", True),
    auto_promotion_min_labels=_get_int("AUTO_PROMOTION_MIN_LABELS", 120),
    auto_promotion_min_stressed_signed_bps=_get_float("AUTO_PROMOTION_MIN_STRESSED_SIGNED_BPS", 0.0),
    auto_promotion_require_oos_positive=_get_bool("AUTO_PROMOTION_REQUIRE_OOS_POSITIVE", True),
    model_monitoring_enabled=_get_bool("MODEL_MONITORING_ENABLED", True),
    model_monitor_lookback=_get_int("MODEL_MONITOR_LOOKBACK", 10000),
    model_monitor_horizon_minutes=_get_int("MODEL_MONITOR_HORIZON_MINUTES", 15),
    model_monitor_min_confidence=_get_float("MODEL_MONITOR_MIN_CONFIDENCE", 0.0),
    model_monitor_short_window_days=_get_int("MODEL_MONITOR_SHORT_WINDOW_DAYS", 7),
    model_monitor_long_window_days=_get_int("MODEL_MONITOR_LONG_WINDOW_DAYS", 30),
    model_monitor_min_labels=_get_int("MODEL_MONITOR_MIN_LABELS", 40),
    model_monitor_min_accuracy_delta=_get_float("MODEL_MONITOR_MIN_ACCURACY_DELTA", -0.07),
    model_monitor_min_signed_bps_delta=_get_float("MODEL_MONITOR_MIN_SIGNED_BPS_DELTA", -8.0),
    model_monitor_max_brier_delta=_get_float("MODEL_MONITOR_MAX_BRIER_DELTA", 0.08),
    model_monitor_breach_streak=_get_int("MODEL_MONITOR_BREACH_STREAK", 3),
    model_monitor_safe_mode_enabled=_get_bool("MODEL_MONITOR_SAFE_MODE_ENABLED", True),
    model_monitor_state_path=os.getenv("MODEL_MONITOR_STATE_PATH", "data/model_monitor_state.json"),
    max_position_age_minutes=_get_int("MAX_POSITION_AGE_MINUTES", 180),
    broker_sync_stale_seconds=_get_int("BROKER_SYNC_STALE_SECONDS", 180),
    stale_position_force_close=_get_bool("STALE_POSITION_FORCE_CLOSE", True),
    require_broker_protection_orders=_get_bool("REQUIRE_BROKER_PROTECTION_ORDERS", True),
    protection_check_grace_seconds=_get_int("PROTECTION_CHECK_GRACE_SECONDS", 20),
    cost_model_enabled=_get_bool("COST_MODEL_ENABLED", False),
    cost_slippage_bps_per_side=_get_float("COST_SLIPPAGE_BPS_PER_SIDE", 1.5),
    cost_fee_per_share=_get_float("COST_FEE_PER_SHARE", 0.0035),
    cost_min_fee_per_order=_get_float("COST_MIN_FEE_PER_ORDER", 0.35),
    cost_apply_in_paper=_get_bool("COST_APPLY_IN_PAPER", True),
    cost_apply_in_mock=_get_bool("COST_APPLY_IN_MOCK", True),
    automation_quality_checks_enabled=_get_bool("AUTOMATION_QUALITY_CHECKS_ENABLED", True),
    automation_quality_check_interval_seconds=_get_int("AUTOMATION_QUALITY_CHECK_INTERVAL_SECONDS", 3600),
    automation_quality_window_minutes=_get_int("AUTOMATION_QUALITY_WINDOW_MINUTES", 60),
    automation_quality_min_samples=_get_int("AUTOMATION_QUALITY_MIN_SAMPLES", 20),
    automation_quality_min_samples_offsession_factor=_get_float("AUTOMATION_QUALITY_MIN_SAMPLES_OFFSESSION_FACTOR", 0.4),
    automation_quality_min_samples_floor=_get_int("AUTOMATION_QUALITY_MIN_SAMPLES_FLOOR", 5),
    automation_quality_enforce_volume_offsession=_get_bool("AUTOMATION_QUALITY_ENFORCE_VOLUME_OFFSESSION", False),
    automation_quality_max_stale_quote_ratio=_get_float("AUTOMATION_QUALITY_MAX_STALE_QUOTE_RATIO", 0.20),
    automation_quality_max_missing_forecast_ratio=_get_float("AUTOMATION_QUALITY_MAX_MISSING_FORECAST_RATIO", 0.10),
    automation_quality_max_bad_sample_ratio=_get_float("AUTOMATION_QUALITY_MAX_BAD_SAMPLE_RATIO", 0.25),
    automation_quality_throttle_cycle_seconds=_get_int("AUTOMATION_QUALITY_THROTTLE_CYCLE_SECONDS", 60),
    sample_flow_guard_enabled=_get_bool("SAMPLE_FLOW_GUARD_ENABLED", True),
    sample_flow_guard_stall_minutes_in_session=_get_int("SAMPLE_FLOW_GUARD_STALL_MINUTES_IN_SESSION", 12),
    automation_cost_guard_enabled=_get_bool("AUTOMATION_COST_GUARD_ENABLED", True),
    automation_cost_daily_budget_usd=_get_float("AUTOMATION_COST_DAILY_BUDGET_USD", 8.0),
    automation_cost_warn_fraction=_get_float("AUTOMATION_COST_WARN_FRACTION", 0.75),
    automation_cost_hard_block=_get_bool("AUTOMATION_COST_HARD_BLOCK", True),
    automation_cost_est_primary_call_usd=_get_float("AUTOMATION_COST_EST_PRIMARY_CALL_USD", 0.0012),
    automation_cost_est_secondary_call_usd=_get_float("AUTOMATION_COST_EST_SECONDARY_CALL_USD", 0.0060),
    automation_cost_mode_enabled=_get_bool("AUTOMATION_COST_MODE_ENABLED", True),
    automation_cost_mode_cycle_seconds=_get_int("AUTOMATION_COST_MODE_CYCLE_SECONDS", 90),
    automation_cost_mode_skip_low_volume_collect=_get_bool("AUTOMATION_COST_MODE_SKIP_LOW_VOLUME_COLLECT", True),
    automation_cost_mode_disable_finnhub_context=_get_bool("AUTOMATION_COST_MODE_DISABLE_FINNHUB_CONTEXT", True),
    automation_cost_mode_disable_economic_calendar=_get_bool("AUTOMATION_COST_MODE_DISABLE_ECONOMIC_CALENDAR", True),
    budget_lock_enabled=_get_bool("BUDGET_LOCK_ENABLED", True),
    budget_lock_in_session_only=_get_bool("BUDGET_LOCK_IN_SESSION_ONLY", True),
    budget_lock_max_llm_calls_per_hour=_get_int("BUDGET_LOCK_MAX_LLM_CALLS_PER_HOUR", 24),
    budget_lock_state_change_min_seconds=_get_int("BUDGET_LOCK_STATE_CHANGE_MIN_SECONDS", 600),
    budget_lock_disable_secondary_escalation=_get_bool("BUDGET_LOCK_DISABLE_SECONDARY_ESCALATION", True),
    automation_auto_recovery_enabled=_get_bool("AUTOMATION_AUTO_RECOVERY_ENABLED", True),
    automation_auto_recovery_cooldown_seconds=_get_int("AUTOMATION_AUTO_RECOVERY_COOLDOWN_SECONDS", 300),
    robust_allowlist_enabled=_get_bool("ROBUST_ALLOWLIST_ENABLED", True),
    robust_allowlist_horizon_minutes=_get_int("ROBUST_ALLOWLIST_HORIZON_MINUTES", 15),
    robust_allowlist_min_labels=_get_int("ROBUST_ALLOWLIST_MIN_LABELS", 20),
    robust_allowlist_n_bootstrap=_get_int("ROBUST_ALLOWLIST_N_BOOTSTRAP", 300),
    robust_allowlist_cost_stress_multiplier=_get_float("ROBUST_ALLOWLIST_COST_STRESS_MULTIPLIER", 1.5),
    robust_allowlist_require_oos_positive=_get_bool("ROBUST_ALLOWLIST_REQUIRE_OOS_POSITIVE", True),
    robust_policy_tiering_enabled=_get_bool("ROBUST_POLICY_TIERING_ENABLED", True),
    robust_policy_tier_coverage_hours_strict=_get_float("ROBUST_POLICY_TIER_COVERAGE_HOURS_STRICT", 12.0),
    robust_policy_tier_coverage_hours_balanced=_get_float("ROBUST_POLICY_TIER_COVERAGE_HOURS_BALANCED", 6.0),
    robust_policy_tier_strict_min_labels=_get_int("ROBUST_POLICY_TIER_STRICT_MIN_LABELS", 24),
    robust_policy_tier_balanced_min_labels=_get_int("ROBUST_POLICY_TIER_BALANCED_MIN_LABELS", 16),
    robust_policy_tier_explore_min_labels=_get_int("ROBUST_POLICY_TIER_EXPLORE_MIN_LABELS", 10),
    robust_policy_tier_strict_cost_stress_multiplier=_get_float("ROBUST_POLICY_TIER_STRICT_COST_STRESS_MULTIPLIER", 1.75),
    robust_policy_tier_balanced_cost_stress_multiplier=_get_float("ROBUST_POLICY_TIER_BALANCED_COST_STRESS_MULTIPLIER", 1.35),
    robust_policy_tier_explore_cost_stress_multiplier=_get_float("ROBUST_POLICY_TIER_EXPLORE_COST_STRESS_MULTIPLIER", 1.10),
    cell_cooldown_enabled=_get_bool("CELL_COOLDOWN_ENABLED", True),
    cell_cooldown_minutes=_get_int("CELL_COOLDOWN_MINUTES", 180),
    cell_cooldown_min_labels=_get_int("CELL_COOLDOWN_MIN_LABELS", 8),
    cell_cooldown_signed_bps_kill=_get_float("CELL_COOLDOWN_SIGNED_BPS_KILL", -20.0),
    symbol_quarantine_enabled=_get_bool("SYMBOL_QUARANTINE_ENABLED", True),
    symbol_quarantine_horizon_minutes=_get_int("SYMBOL_QUARANTINE_HORIZON_MINUTES", 15),
    symbol_quarantine_min_labels=_get_int("SYMBOL_QUARANTINE_MIN_LABELS", 30),
    symbol_quarantine_signed_bps_kill=_get_float("SYMBOL_QUARANTINE_SIGNED_BPS_KILL", -15.0),
    symbol_quarantine_minutes=_get_int("SYMBOL_QUARANTINE_MINUTES", 240),
    symbol_quarantine_recover_min_labels=_get_int("SYMBOL_QUARANTINE_RECOVER_MIN_LABELS", 20),
    symbol_quarantine_recover_signed_bps=_get_float("SYMBOL_QUARANTINE_RECOVER_SIGNED_BPS", 2.0),
    autonomous_research_enabled=_get_bool("AUTONOMOUS_RESEARCH_ENABLED", True),
    autonomous_research_check_seconds=_get_int("AUTONOMOUS_RESEARCH_CHECK_SECONDS", 300),
    autonomous_research_suite_interval_minutes=_get_int("AUTONOMOUS_RESEARCH_SUITE_INTERVAL_MINUTES", 60),
    autonomous_self_scan_enabled=_get_bool("AUTONOMOUS_SELF_SCAN_ENABLED", True),
    autonomous_self_scan_interval_minutes=_get_int("AUTONOMOUS_SELF_SCAN_INTERVAL_MINUTES", 30),
    autonomous_self_scan_lookback=_get_int("AUTONOMOUS_SELF_SCAN_LOOKBACK", 2000),
    audit_log_path=os.getenv("AUDIT_LOG_PATH", "data/audit_events.jsonl"),
    app_db_path=os.getenv("APP_DB_PATH", "data/trading_app.db"),
    database_url=os.getenv("DATABASE_URL", ""),
    app_auth_enabled=_get_bool("APP_AUTH_ENABLED", False),
    app_auth_username=os.getenv("APP_AUTH_USERNAME", "admin"),
    app_auth_password=os.getenv("APP_AUTH_PASSWORD", ""),
    notifications_enabled=_get_bool("NOTIFICATIONS_ENABLED", True),
    notifications_path=os.getenv("NOTIFICATIONS_PATH", "data/notifications.jsonl"),
    api_host=os.getenv("API_HOST", "127.0.0.1"),
    api_port=_get_int("API_PORT", 8000),
)


def load_settings() -> Settings:
    _migrate_src_data_to_data()
    return replace(
        DEFAULT_SETTINGS,
        edge_monitor_state_path=_normalize_runtime_path(DEFAULT_SETTINGS.edge_monitor_state_path),
        macro_snapshot_path=_normalize_runtime_path(DEFAULT_SETTINGS.macro_snapshot_path),
        adaptive_state_path=_normalize_runtime_path(DEFAULT_SETTINGS.adaptive_state_path),
        auto_retrain_state_path=_normalize_runtime_path(DEFAULT_SETTINGS.auto_retrain_state_path),
        promotion_state_path=_normalize_runtime_path(DEFAULT_SETTINGS.promotion_state_path),
        model_monitor_state_path=_normalize_runtime_path(DEFAULT_SETTINGS.model_monitor_state_path),
        audit_log_path=_normalize_runtime_path(DEFAULT_SETTINGS.audit_log_path),
        app_db_path=_normalize_runtime_path(DEFAULT_SETTINGS.app_db_path),
        notifications_path=_normalize_runtime_path(DEFAULT_SETTINGS.notifications_path),
    )
