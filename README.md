# AI Trading Engine

Application version of the trading system with:
- FastAPI control API + browser dashboard
- Background worker loop
- Persistent storage (SQLite or Postgres)
- Pluggable providers (mock or Alpaca paper)

## Local run

From project root:
- `$env:PYTHONPATH='src'`
- `python -m ai_trading_engine.server`

Open: `http://127.0.0.1:8000`

Worker options:
- one-shot cycle via API: `POST /api/run-once`
- dedicated worker process: `python -m ai_trading_engine.worker`
- auto-start on API boot: set `AUTO_START_WORKER=true` in `.env`

## API

- `GET /api/status`
- `POST /api/start`
- `POST /api/stop`
- `POST /api/run-once`
- `GET /api/account`
- `GET /api/config`
- `GET /api/adaptive`
- `GET /api/acceleration`
- `POST /api/acceleration/mode?mode=standard|accelerated`
- `GET /api/metrics`
- `POST /api/retrain?lookback=2000`
- `POST /api/research/walk-forward?lookback=10000&folds=4&min_train=40&min_test=20&bins=10`
- `POST /api/research/predictive?lookback=10000&folds=4&min_train=40&min_test=20&n_estimators=80&learning_rate=0.1&max_bins=16`
- `GET /api/automation`
- `GET /api/promotion/status`
- `POST /api/promotion/evaluate`
- `POST /api/promotion/promote`
- `GET /api/go-live-gate`
- `GET /api/audit?limit=50`
- `GET /api/decisions?limit=20`
- `GET /api/trades?limit=20`

Retraining:
- Rebuild adaptive learner state from persisted closed trades:
  - `POST /api/retrain?lookback=2000`
- `lookback` controls how many most-recent closed trades are used (max `50000`).

Research:
- Run walk-forward threshold calibration from persisted closed trades:
  - `POST /api/research/walk-forward?lookback=10000&folds=4&min_train=40&min_test=20&bins=10`
- Run predictive model walk-forward (gradient-boosted stumps + isotonic calibration):
  - `POST /api/research/predictive?lookback=10000&folds=4&min_train=40&min_test=20&n_estimators=80&learning_rate=0.1&max_bins=16`
- Report output path:
  - `data/research/walk_forward_report_*.json`
  - `data/research/predictive_walk_forward_*.json`

Automation:
- Enable scheduled retraining/research:
  - `AUTO_RETRAIN_ENABLED=true`
- Default cadence:
  - `AUTO_RETRAIN_INTERVAL_HOURS=168` (weekly)
  - `AUTO_RETRAIN_MIN_NEW_TRADES=50`
  - `AUTO_RETRAIN_NEG_EXPECTANCY_LOOKBACK=30` (trigger if rolling expectancy < 0)
  - `AUTO_RETRAIN_CHECK_SECONDS=900`
- Optional research job after retrain:
  - `AUTO_RESEARCH_ENABLED=true`
  - `AUTO_RESEARCH_LOOKBACK=10000`
  - `AUTO_RESEARCH_FOLDS=4`
  - `AUTO_RESEARCH_MIN_TRAIN=40`
  - `AUTO_RESEARCH_MIN_TEST=20`
  - `AUTO_RESEARCH_BINS=10`
- Automation state file:
  - `AUTO_RETRAIN_STATE_PATH=data/automation_state.json`

Promotion policy:
- Evaluate latest predictive report against strict criteria:
  - `POST /api/promotion/evaluate`
- Promote only when all checks pass:
  - `POST /api/promotion/promote`
- Promotion state file:
  - `PROMOTION_STATE_PATH=data/promotion_state.json`
- Default strict criteria:
  - `PROMOTION_MIN_FOLDS=3`
  - `PROMOTION_MIN_MODEL_SELECTED_TRADES=40`
  - `PROMOTION_MIN_EXPECTANCY=0.0`
  - `PROMOTION_MIN_NET_PNL_EDGE=0.0`
  - `PROMOTION_REQUIRE_RECOMMENDATION_PROMOTE=true`

## Docker (API + Worker + Postgres)

1. Copy env: `Copy-Item .env.example .env`
2. Set provider/env keys in `.env` as needed.
3. Start stack: `docker compose up --build`
4. Open app: `http://localhost:8000`

Compose services:
- `db` (Postgres 16)
- `api` (FastAPI dashboard/API)
- `worker` (continuous trading cycles)

## Railway split services

- The container entrypoint supports `APP_ROLE=api` or `APP_ROLE=worker`.
- For the web service (`apex-01`):
  - keep the health check on `/health`
  - set `APP_ROLE=api`
  - set `AUTO_START_WORKER=false` so the API process does not launch its own in-process worker
- For the worker service:
  - create a second Railway service from the same repo/image
  - set `APP_ROLE=worker`
  - do not configure an HTTP health check or public domain
  - point it at the same `/data` volume and the same runtime env as the web service
- This lets Railway run a dedicated worker process with the same codebase and environment, while keeping the web service focused on the API only.

## Providers

- Default safe mode:
  - `DATA_PROVIDER=mock`
  - `EXECUTION_PROVIDER=mock`
- Alpaca paper mode:
  - `DATA_PROVIDER=alpaca`
  - `EXECUTION_PROVIDER=alpaca`
  - set `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY`
  - use equity symbols supported by Alpaca (e.g., `SPY`)

## Adaptive learning

- Enable with `ADAPTIVE_ENABLED=true`
- State file path via `ADAPTIVE_STATE_PATH` (default `data/adaptive_state.json`)
- Learns online from closed trades by regime/direction and confidence bins
- Dynamically adjusts confidence and size; can auto-hold weak regime-direction combinations

## Autonomous live gate

- Autonomous live worker start is blocked unless go-live criteria pass.
- The guard applies when:
  - `EXECUTION_PROVIDER=alpaca`
  - `ALPACA_TRADING_URL` points to live endpoint (not paper)
- Endpoint:
  - `GET /api/go-live-gate`
- Enable autonomous live only after passing:
  - `AUTONOMOUS_LIVE_ENABLED=true`
- Criteria knobs:
  - `GO_LIVE_MIN_CLOSED_TRADES`
  - `GO_LIVE_MIN_WIN_RATE`
  - `GO_LIVE_MIN_PROFIT_FACTOR`
  - `GO_LIVE_MIN_EXPECTANCY`
  - `GO_LIVE_MAX_DRAWDOWN_PCT`
  - `GO_LIVE_MAX_HOLD_RATE`
  - `GO_LIVE_METRICS_LOOKBACK`

## Pre-trade risk gateway + audit trail

- Every trade submission is evaluated by a dedicated pre-trade risk gateway.
- Rejected orders are converted to `hold` with explicit reason.
- Gateway checks:
  - max single-order notional
  - max total exposure after order
  - max quote age (stale quote guard)
  - max pre-trade spread
  - minimum buying power
  - duplicate order cooldown
- Controls:
  - `PRETRADE_MAX_ORDER_NOTIONAL_USD`
  - `PRETRADE_MAX_TOTAL_EXPOSURE_USD`
  - `PRETRADE_MAX_QUOTE_AGE_SECONDS`
  - `PRETRADE_MAX_SPREAD_BPS`
  - `PRETRADE_MIN_BUYING_POWER_USD`
  - `PRETRADE_DUPLICATE_COOLDOWN_SECONDS`
- Append-only audit log:
  - `AUDIT_LOG_PATH`
  - hash-chained JSONL events for `pretrade_check`, `order_submitted`, and `order_submit_error`

## Costs (fees + slippage)

- Apply conservative round-trip execution costs to closed-trade PnL before learning/research metrics.
- Controls:
  - `COST_MODEL_ENABLED`
  - `COST_SLIPPAGE_BPS_PER_SIDE`
  - `COST_FEE_PER_SHARE`
  - `COST_MIN_FEE_PER_ORDER`
  - `COST_APPLY_IN_PAPER`
  - `COST_APPLY_IN_MOCK`
- Live safeguard:
  - Cost model is not applied when Alpaca live endpoint is detected.

## Entry quality gates

- Session window filter:
  - `ENABLE_SESSION_FILTER=true`
  - `SESSION_START_ET=09:35`
  - `SESSION_END_ET=15:45`
- Spread filter:
  - `MAX_SPREAD_BPS=8.0`
- Confluence filter:
  - `ENTRY_CONFLUENCE_MIN=58.0`
- EV gate:
  - `EV_MIN_TICKS=0.2`
- ATR risk sizing:
  - `PRICE_TICK_SIZE=0.01`
  - `RISK_PER_TRADE_USD=75.0`
- Trades are converted to `hold` when any gate fails.

## Data Acceleration Mode

- Purpose: increase labeled sample throughput in paper mode while preserving hard risk and pre-trade controls.
- Enable:
  - `DATA_ACCELERATION_MODE=true`
- Relaxation knobs:
  - `ACCEL_RELAX_CONFLUENCE` (subtract from `ENTRY_CONFLUENCE_MIN`)
  - `ACCEL_RELAX_EV_TICKS` (subtract from `EV_MIN_TICKS`)
  - `ACCEL_SPREAD_MULT` (multiplier on `MAX_SPREAD_BPS`)
  - `ACCEL_SESSION_START_ET`
  - `ACCEL_SESSION_END_ET`
  - `ACCEL_EDGE_MIN_WIN_RATE_DELTA` (subtract from `EDGE_MIN_WIN_RATE`)
  - `ACCEL_EDGE_MIN_EXPECTANCY_DELTA` (subtract from `EDGE_MIN_EXPECTANCY`)
- Visibility:
  - `GET /api/acceleration`
- Safety:
  - Automatically disabled when Alpaca live endpoint is detected (`ALPACA_TRADING_URL` not paper).

## Anti-decay monitor

- Rolling edge monitor + auto-throttle:
  - `EDGE_WINDOW_TRADES=40`
  - `EDGE_MIN_WIN_RATE=0.42`
  - `EDGE_MIN_EXPECTANCY=0.0`
  - `EDGE_THROTTLE_SIZE_MULT=0.6`
  - `EDGE_MONITOR_STATE_PATH=data/edge_monitor_state.json`
- Champion/Challenger shadow tracking:
  - `SHADOW_CHALLENGER_ENABLED=true`
  - `SHADOW_CONFLUENCE_BUMP=6.0`
  - `SHADOW_EV_BUMP=0.4`

## Persistence

- Local SQLite: set `APP_DB_PATH`, leave `DATABASE_URL` empty.
- Postgres: set `DATABASE_URL` (compose sets this automatically).

## Tests

- `$env:PYTHONPATH='src'`
- `python -m unittest discover -s tests -v`

## Historical backtest (Alpaca bars)

Runs a bar-replay backtest using Alpaca historical data with your current strategy/risk settings.

- Requires `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` in `.env`
- Command:
  - `$env:PYTHONPATH='src'`
  - `python -m ai_trading_engine.historical_backtest --symbol SPY --start 2026-03-01 --end 2026-04-20 --starting-balance 100000`
- Output report:
  - `data/backtests/historical_backtest_*.json`

## Safety

This remains development-grade software.
Use paper trading first; add authentication, reconciliation, broker error handling, and kill-switch logic before live deployment.

## App Access Control

- Optional single-user Basic Auth for all routes (UI + API):
  - `APP_AUTH_ENABLED=true`
  - `APP_AUTH_USERNAME=<your-username>`
  - `APP_AUTH_PASSWORD=<strong-password>`
- When enabled, browser/API requests must provide credentials.

## Decision Metadata + Notifications

- New decisions and closed trades include metadata with:
  - model name/version
  - active mode and effective thresholds
  - cost model settings
  - data/execution/LLM providers
- Local notification feed:
  - `GET /api/notifications?limit=50`
  - `NOTIFICATIONS_ENABLED=true`
  - `NOTIFICATIONS_PATH=data/notifications.jsonl`
- Current events include worker start/stop/failure, trade placed, and position closed.
