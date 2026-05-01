@echo off
cd /d C:\Users\yxngh\Documents\ai_trading_engine
set PYTHONPATH=src

:loop
C:\Users\yxngh\AppData\Local\Programs\Python\Python313\python.exe -m ai_trading_engine.server
timeout /t 5 /nobreak >nul
goto loop
