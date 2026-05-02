#!/bin/sh
set -eu

DB_PATH="${APP_DB_PATH:-/data/trading_app.db}"
SEED_PATH="/app/seed/trading_app.db"
FORCE_DB_SEED="${FORCE_DB_SEED:-false}"

DB_DIR="$(dirname "$DB_PATH")"
mkdir -p "$DB_DIR"

if [ "$FORCE_DB_SEED" = "true" ] && [ -f "$SEED_PATH" ]; then
  cp "$SEED_PATH" "$DB_PATH"
  echo "Force-seeded database at $DB_PATH from $SEED_PATH"
elif [ -f "$SEED_PATH" ] && { [ ! -f "$DB_PATH" ] || [ ! -s "$DB_PATH" ]; }; then
  cp "$SEED_PATH" "$DB_PATH"
  echo "Seeded database at $DB_PATH from $SEED_PATH"
fi

exec python -m uvicorn ai_trading_engine.api:app --app-dir src --host 0.0.0.0 --port "${PORT:-8000}"
