$ErrorActionPreference = "Stop"
Set-Location "C:\Users\yxngh\Documents\ai_trading_engine"
$env:PYTHONPATH = "src"
& "C:\Users\yxngh\AppData\Local\Programs\Python\Python313\python.exe" -m ai_trading_engine.server
