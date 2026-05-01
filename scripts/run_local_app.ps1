$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Resolve-Path "$PSScriptRoot\..")
$env:PYTHONPATH = "src"
python -m uvicorn ai_trading_engine.api:app --host 127.0.0.1 --port 8000
