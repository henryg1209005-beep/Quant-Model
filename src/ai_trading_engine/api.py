from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse, Response

from ai_trading_engine.app_service import TradingAppService
from ai_trading_engine.auth import verify_basic_auth
from ai_trading_engine.config import load_settings
from ai_trading_engine.metadata import MODEL_NAME
from ai_trading_engine.persistence import Persistence

settings = load_settings()
persistence = Persistence(settings.app_db_path, database_url=settings.database_url)
service = TradingAppService(settings, persistence)

app = FastAPI(title=f"{MODEL_NAME} App", version="0.1.0")


@app.middleware("http")
async def auth_guard(request: Request, call_next):
    if request.url.path in {"/health"}:
        return await call_next(request)
    if not settings.app_auth_enabled:
        return await call_next(request)
    if verify_basic_auth(
        request.headers.get("authorization"),
        username=settings.app_auth_username,
        password=settings.app_auth_password,
    ):
        return await call_next(request)
    return JSONResponse(
        status_code=401,
        content={"detail": "Authentication required"},
        headers={"WWW-Authenticate": f'Basic realm="{MODEL_NAME}"'},
    )


@app.on_event("startup")
def app_startup() -> None:
    if settings.auto_start_worker:
        service.start_with_guard()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Apex .01</title>
  <style>
    :root {
      --bg: #f3f4f6;
      --text: #0d1117;
      --muted: #6b7280;
      --surface: #ffffff;
      --surface-soft: #f9fafb;
      --line: #e1e4ea;
      --line-strong: #c4cad4;
      --brand: #1a56db;
      --brand-2: #1e40af;
      --danger: #b42318;
      --warn: #b45309;
      --ok: #166534;
      --ink: #0d1117;
      --radius: 3px;
      --shadow: 0 1px 3px rgba(15,23,42,0.07), 0 0 0 1px rgba(15,23,42,0.04);
      --shadow-soft: 0 1px 2px rgba(15,23,42,0.05);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      font-size: 13px;
      line-height: 1.5;
    }
    .app {
      max-width: 1500px;
      margin: 0 auto;
      padding: 16px 20px;
    }
    .header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
      padding: 16px 20px;
      color: #e2e8f0;
      background: #0d1117;
      border: 1px solid #1e2733;
      border-radius: var(--radius);
    }
    h1 {
      margin: 0;
      font-size: 17px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #f1f5f9;
    }
    .subtitle {
      margin: 4px 0 0;
      color: #64748b;
      font-size: 11px;
      font-weight: 400;
      letter-spacing: 0.01em;
    }
    .chips {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .chip {
      border: 1px solid #2d3a4a;
      background: #141d2b;
      border-radius: 2px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 600;
      color: #7a8fa8;
      letter-spacing: 0.03em;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .chip.ok   { border-color: #1a4731; background: #0a2318; color: #6ee7b7; }
    .chip.warn { border-color: #78350f; background: #1c0f05; color: #fbbf24; }
    .chip.danger { border-color: #7f1d1d; background: #1a0808; color: #fca5a5; }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
      gap: 6px;
      margin-bottom: 6px;
    }
    .toolbar.secondary {
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      margin-top: 0;
      margin-bottom: 8px;
    }
    button {
      border: 1px solid rgba(15,23,42,0.14);
      border-radius: var(--radius);
      padding: 8px 12px;
      color: #fff;
      font-weight: 600;
      font-size: 12px;
      cursor: pointer;
      letter-spacing: 0.01em;
      transition: opacity .1s ease;
    }
    button:hover { opacity: 0.82; }
    button:active { opacity: 0.65; }
    button.start  { background: #166534; border-color: #14532d; }
    button.stop   { background: var(--danger); border-color: #991b1b; }
    button.once   { background: var(--brand); border-color: #1e40af; }
    button.refresh {
      color: var(--text);
      border-color: var(--line-strong);
      background: var(--surface);
    }
    button.research {
      color: #1e293b;
      border-color: #c4cad4;
      background: #f8fafc;
    }
    .flash {
      min-height: 20px;
      margin: 4px 0 8px;
      color: #374151;
      font-size: 12px;
      font-style: italic;
    }
    .opsline {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 6px;
      margin-bottom: 14px;
    }
    .opsitem {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: var(--radius);
      padding: 8px 12px;
      font-size: 12px;
      color: #1e293b;
    }
    .opsitem strong {
      display: block;
      color: var(--muted);
      font-weight: 700;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 3px;
    }
    .blocker { display: block; margin-top: 2px; line-height: 1.3; }
    .blocker-label {
      display: inline-block;
      border-radius: 2px;
      padding: 2px 7px;
      background: #fff7ed;
      border: 1px solid #fdba74;
      color: #9a3412;
      font-weight: 700;
      font-size: 11px;
      margin-bottom: 3px;
    }
    .blocker-detail {
      display: block;
      color: #4b5563;
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .tabs {
      display: flex;
      gap: 0;
      margin-bottom: 16px;
      border-bottom: 2px solid var(--line);
    }
    .tabbtn {
      border: none;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      border-radius: 0;
      background: transparent;
      color: var(--muted);
      padding: 8px 18px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      letter-spacing: 0.01em;
      transition: color .1s ease;
    }
    .tabbtn.active {
      background: transparent;
      border-bottom-color: var(--ink);
      color: var(--ink);
    }
    .tabbtn:hover:not(.active) { color: var(--text); }
    .panel { display: none; }
    .panel.active { display: block; }
    .subtabs {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-bottom: 14px;
    }
    .subtabbtn {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      color: #374151;
      padding: 5px 10px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      transition: background .1s ease, color .1s ease;
    }
    .subtabbtn.active {
      background: var(--ink);
      border-color: var(--ink);
      color: #f1f5f9;
    }
    .analytics-panel { display: none; }
    .analytics-panel.active { display: block; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow-soft);
      padding: 14px;
    }
    .kpis .card {
      border-top: 2px solid var(--line-strong);
    }
    .kpi-label {
      font-size: 10px;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }
    .kpi-val {
      margin-top: 6px;
      font-size: 21px;
      font-weight: 700;
      line-height: 1.15;
      color: var(--ink);
      font-variant-numeric: tabular-nums;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }
    @media (min-width: 980px) {
      .grid { grid-template-columns: 1fr 1fr; }
      .span-2 { grid-column: 1 / -1; }
    }
    .card h3 {
      margin: 0 0 12px;
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }
    .kv {
      display: grid;
      grid-template-columns: 158px minmax(0, 1fr);
      gap: 5px 10px;
      font-size: 12px;
    }
    .k { color: var(--muted); font-weight: 600; font-size: 11px; }
    .v {
      border-bottom: 1px solid #f0f2f6;
      padding-bottom: 5px;
      word-break: break-word;
      font-size: 12px;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 740px;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      padding: 7px 10px;
      border-bottom: 1px solid #eef0f5;
      vertical-align: middle;
    }
    th {
      position: sticky;
      top: 0;
      background: #f3f4f6;
      color: #6b7280;
      z-index: 1;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      font-weight: 700;
      border-bottom: 1px solid var(--line);
    }
    tr:hover td { background: #fafbfc; }
    .badge {
      display: inline-block;
      border-radius: 2px;
      border: 1px solid #d1d5db;
      padding: 2px 6px;
      font-size: 10px;
      font-weight: 700;
      background: #f9fafb;
      color: #374151;
      white-space: nowrap;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .b-long  { border-color: #86efac; background: #f0fdf4; color: #166534; }
    .b-short { border-color: #fca5a5; background: #fef2f2; color: #991b1b; }
    .b-trade { border-color: #67e8f9; background: #ecfeff; color: #0e7490; }
    .b-hold  { border-color: #fcd34d; background: #fffbeb; color: #92400e; }
    .small { color: var(--muted); font-size: 11px; }
    .ellipsis {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      max-width: 100%;
      display: block;
    }
    .analytics-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }
    @media (min-width: 980px) {
      .analytics-grid { grid-template-columns: 1fr 1fr 1fr; }
    }
    .chart-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      padding: 12px;
    }
    .chart-title {
      font-size: 10px;
      color: var(--muted);
      margin-bottom: 8px;
      font-weight: 700;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }
    .chart-svg {
      width: 100%;
      height: 200px;
      display: block;
      background: #f9fafb;
      border: 1px solid #eef0f5;
      border-radius: var(--radius);
    }
    .chart-empty {
      font-size: 12px;
      color: var(--muted);
      padding: 64px 8px;
      text-align: center;
      background: #f9fafb;
      border: 1px dashed var(--line-strong);
      border-radius: var(--radius);
    }
    @media (max-width: 720px) {
      .app { padding: 10px 12px; }
      .header { padding: 12px 14px; }
      h1 { font-size: 15px; }
      .toolbar, .toolbar.secondary { grid-template-columns: 1fr 1fr; }
      .kv { grid-template-columns: 1fr; }
      .k { padding-top: 3px; }
      .tabs { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div>
        <h1>Apex .01</h1>
        <p class="subtitle">Operational dashboard for status, risk gates, execution, audit trail, and visual analytics</p>
      </div>
      <div class="chips">
        <div id="chipWorker" class="chip">Worker: -</div>
        <div id="chipGoLive" class="chip">Go-Live: -</div>
        <div id="chipMode" class="chip">Mode: -</div>
      </div>
    </div>

    <div class="toolbar">
      <button class="start" onclick="post('/api/start', 'Start requested')">Start Worker</button>
      <button class="stop" onclick="post('/api/stop', 'Stop requested')">Stop Worker</button>
      <button class="once" onclick="post('/api/run-once', 'Single cycle requested')">Run One Cycle</button>
      <button class="refresh" onclick="refreshAll()">Refresh Now</button>
    </div>
    <div class="toolbar secondary">
      <button class="research" onclick="runResearch('/api/retrain?lookback=5000', 'Retrain triggered')">Run Retrain</button>
      <button class="research" onclick="runResearch('/api/research/walk-forward?lookback=10000&folds=4&min_train=40&min_test=20&bins=10', 'Walk-forward report generated')">Run Walk-Forward</button>
      <button class="research" onclick="runResearch('/api/research/predictive?lookback=10000&folds=4&min_train=40&min_test=20&n_estimators=80&learning_rate=0.1&max_bins=16', 'Predictive report generated')">Run Predictive Research</button>
      <button class="research" onclick="setAccelerationMode('standard')">Mode: Standard</button>
      <button class="research" onclick="setAccelerationMode('accelerated')">Mode: Accelerated</button>
    </div>
    <div id="flash" class="flash"></div>
    <div class="opsline">
      <div class="opsitem"><strong>Last Refresh:</strong> <span id="opsRefresh">-</span></div>
      <div class="opsitem"><strong>NY Session:</strong> <span id="opsSession">-</span></div>
      <div class="opsitem"><strong>Current Blocker:</strong> <span id="opsBlocker" class="blocker">-</span></div>
      <div class="opsitem"><strong>System Health:</strong> <span id="opsHealth">-</span></div>
    </div>
    <div class="tabs">
      <button id="tabOverviewBtn" class="tabbtn active" onclick="showTab('overview')">Overview</button>
      <button id="tabAnalyticsBtn" class="tabbtn" onclick="showTab('analytics')">Analytics</button>
    </div>

    <div id="panelOverview" class="panel active">

      <div class="kpi-section">
        <div class="kpi-section-label">Trading</div>
        <div class="kpis">
          <div class="card"><div class="kpi-label">Balance</div><div id="kBalance" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Daily PnL</div><div id="kPnl" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Trades Today</div><div id="kTradesToday" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Open Position</div><div id="kOpenPos" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Trade Count</div><div id="kTradeCount" class="kpi-val">-</div></div>
        </div>
      </div>

      <div class="kpi-section">
        <div class="kpi-section-label">Performance</div>
        <div class="kpis">
          <div class="card"><div class="kpi-label">Win Rate</div><div id="kWinRate" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Profit Factor</div><div id="kPf" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Cycles</div><div id="kCycles" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Latest Note</div><div id="kNote" class="kpi-val" style="font-size:0.85rem">-</div></div>
        </div>
      </div>

      <div class="kpi-section">
        <div class="kpi-section-label">Data &amp; Health</div>
        <div class="kpis">
          <div class="card"><div class="kpi-label">Autonomy Health</div><div id="kAutonomyHealth" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Data Readiness</div><div id="kDataReadiness" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Data Pace (1h)</div><div id="kDataPace" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Projected Daily Samples</div><div id="kProjSamples" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">LLM Provider Health</div><div id="kLlmHealth" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Top Error Cause</div><div id="kTopCause" class="kpi-val">-</div></div>
        </div>
      </div>

      <div class="kpi-section">
        <div class="kpi-section-label">System</div>
        <div class="kpis">
          <div class="card"><div class="kpi-label">Weekly Experiment</div><div id="kWeeklyExp" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Auto Promotion Gate</div><div id="kAutoPromoGate" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">Sample Flow Guard</div><div id="kSampleFlowGuard" class="kpi-val">-</div></div>
          <div class="card"><div class="kpi-label">7D Sprint</div><div id="kSprint7d" class="kpi-val">-</div></div>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <h3>Status</h3>
          <div id="statusKV" class="kv"></div>
        </div>
        <div class="card">
          <h3>Runtime Config</h3>
          <div id="configKV" class="kv"></div>
        </div>
        <div class="card">
          <h3>Account</h3>
          <div id="accountKV" class="kv"></div>
        </div>
        <div class="card">
          <h3>Go-Live Gate</h3>
          <div id="goLiveKV" class="kv"></div>
        </div>
        <div class="card span-2">
          <h3>Performance Metrics</h3>
          <div id="metricsKV" class="kv"></div>
        </div>
      </div>
    </div>

    <div id="panelAnalytics" class="panel">
      <div class="subtabs">
        <button id="subtabChartsBtn" class="subtabbtn active" onclick="showAnalyticsSubtab('charts')">Charts</button>
        <button id="subtabQualityBtn" class="subtabbtn" onclick="showAnalyticsSubtab('quality')">Prediction Quality</button>
        <button id="subtabInventoryBtn" class="subtabbtn" onclick="showAnalyticsSubtab('inventory')">Feature Inventory</button>
        <button id="subtabPatternsBtn" class="subtabbtn" onclick="showAnalyticsSubtab('patterns')">Pattern Leaderboard</button>
        <button id="subtabContextBtn" class="subtabbtn" onclick="showAnalyticsSubtab('context')">Context Leaderboard</button>
        <button id="subtabSymbolGateBtn" class="subtabbtn" onclick="showAnalyticsSubtab('symbolGate')">Symbol Gate</button>
        <button id="subtabCCBtn" class="subtabbtn" onclick="showAnalyticsSubtab('cc')">Champion vs Challenger</button>
        <button id="subtabDecisionsBtn" class="subtabbtn" onclick="showAnalyticsSubtab('decisions')">Decisions</button>
        <button id="subtabTradesBtn" class="subtabbtn" onclick="showAnalyticsSubtab('trades')">Trades</button>
        <button id="subtabEventsBtn" class="subtabbtn" onclick="showAnalyticsSubtab('events')">Events</button>
      </div>
      <div class="grid">
        <div id="analyticsChartsPanel" class="analytics-panel active span-2">
          <div class="card span-2">
            <h3>Charts</h3>
            <div class="analytics-grid">
              <div class="chart-card">
                <div class="chart-title">Equity Curve</div>
                <div id="eqChart"></div>
              </div>
              <div class="chart-card">
                <div class="chart-title">PnL Distribution</div>
                <div id="pnlHist"></div>
              </div>
              <div class="chart-card">
                <div class="chart-title">Decision Mix</div>
                <div id="decMix"></div>
              </div>
            </div>
          </div>
        </div>

        <div id="analyticsQualityPanel" class="analytics-panel span-2">

          <div class="kpi-section">
            <div class="kpi-section-label">Model Performance</div>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Labelled Predictions</div><div id="qLabels" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Accuracy</div><div id="qAccuracy" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Signed Return</div><div id="qSigned" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Brier Score</div><div id="qBrier" class="kpi-val">-</div></div>
            </div>
          </div>

          <div class="kpi-section">
            <div class="kpi-section-label">Data</div>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Samples With Quote</div><div id="qQuote" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Allowed Regime Cells</div><div id="qAllowedCells" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Best-Horizon Symbols</div><div id="qBestHorizonSymbols" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Sample Skip Rate</div><div id="qSampleSkipRate" class="kpi-val">-</div></div>
            </div>
          </div>

          <div class="kpi-section">
            <div class="kpi-section-label">Calibration &amp; Guards</div>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Coverage Guard</div><div id="qCoverageGuard" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Policy Tier</div><div id="qPolicyTier" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Symbol Quarantine</div><div id="qSymbolQuarantine" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Calibration Gate</div><div id="qCalGate" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Calib Evaluated</div><div id="qCalEval" class="kpi-val">-</div></div>
            </div>
          </div>

          <div class="card" style="margin-bottom:10px">
            <h3>Signed Return by Horizon</h3>
            <div id="qualityChart"></div>
          </div>

          <div class="grid">
            <div class="card">
              <h3>By Symbol (15m)</h3>
              <div class="table-wrap"><table id="qualitySymbolTable"></table></div>
            </div>
            <div class="card">
              <h3>By Confidence Bin (15m)</h3>
              <div class="table-wrap"><table id="qualityConfidenceTable"></table></div>
            </div>
            <div class="card">
              <h3>Calibration Gate by Symbol / Bin</h3>
              <div id="qCalSummary" class="small" style="margin:0 0 8px 0">-</div>
              <div class="table-wrap"><table id="qualityCalibTable"></table></div>
            </div>
            <div class="card">
              <h3>Direction Policy (15m)</h3>
              <div id="qDirectionSummary" class="small" style="margin:0 0 8px 0">-</div>
              <div class="table-wrap"><table id="qualityDirectionPolicyTable"></table></div>
            </div>
          </div>

          <div class="small" id="qualityNote" style="margin-top:10px"></div>
        </div>

        <div id="analyticsInventoryPanel" class="analytics-panel span-2">
          <div class="kpi-section">
            <div class="kpi-section-label">Feature Inventory</div>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Tracked Features</div><div id="fiFeatureCount" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Rows Scanned</div><div id="fiSampleCount" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Core Features</div><div id="fiCoreCount" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Pattern Features</div><div id="fiPatternCount" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Structure Features</div><div id="fiStructureCount" class="kpi-val">-</div></div>
            </div>
          </div>
          <div class="card">
            <h3>Features</h3>
            <div id="fiSummary" class="small" style="margin:0 0 8px 0">-</div>
            <div class="table-wrap"><table id="featureInventoryTable"></table></div>
          </div>
        </div>

        <div id="analyticsPatternsPanel" class="analytics-panel span-2">
          <div class="kpi-section">
            <div class="kpi-section-label">Pattern Leaderboard</div>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Allowed Pattern States</div><div id="plAllowedCount" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Stress Bps</div><div id="plStressBps" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Min Accuracy</div><div id="plMinAccuracy" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Min Labels</div><div id="plMinLabels" class="kpi-val">-</div></div>
            </div>
          </div>
          <div class="card">
            <h3>Pattern States</h3>
            <div id="plSummary" class="small" style="margin:0 0 8px 0">-</div>
            <div class="table-wrap"><table id="patternLeaderboardTable"></table></div>
          </div>
        </div>

        <div id="analyticsContextPanel" class="analytics-panel span-2">
          <div class="kpi-section">
            <div class="kpi-section-label">Context Leaderboard</div>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Allowed Context States</div><div id="clAllowedCount" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Stress Bps</div><div id="clStressBps" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Min Accuracy</div><div id="clMinAccuracy" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Min Labels</div><div id="clMinLabels" class="kpi-val">-</div></div>
            </div>
          </div>
          <div class="card">
            <h3>Context States</h3>
            <div id="clSummary" class="small" style="margin:0 0 8px 0">-</div>
            <div class="table-wrap"><table id="contextLeaderboardTable"></table></div>
          </div>
        </div>

        <div id="analyticsSymbolGatePanel" class="analytics-panel span-2">
          <div class="card span-2">
            <h3>Symbol Gate</h3>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Allowed Symbols</div><div id="sgAllowed" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Blocked Symbols</div><div id="sgBlocked" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Horizon</div><div id="sgHorizon" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Robust Cells</div><div id="sgRobustCells" class="kpi-val">-</div></div>
            </div>
            <div class="table-wrap"><table id="symbolGateTable"></table></div>
            <div class="chart-title" style="margin-top:12px;margin-bottom:8px">Robust Cell Leaderboard (CI95 low &gt; 0)</div>
            <div class="table-wrap"><table id="cellBootstrapTable"></table></div>
          </div>
        </div>

        <div id="analyticsCcPanel" class="analytics-panel span-2">
          <div class="card span-2">
            <h3>Champion vs Challenger (Daily OOS)</h3>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Champion Signed bps</div><div id="ccChampSigned" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Challenger Signed bps</div><div id="ccChallSigned" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Delta Signed bps</div><div id="ccDeltaSigned" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">CI95 (delta)</div><div id="ccCi95" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Significance</div><div id="ccSig" class="kpi-val">-</div></div>
            </div>
            <div class="table-wrap"><table id="ccTable"></table></div>
            <h3 style="margin-top:16px;border-bottom:0;padding-bottom:0">Short Challenger Slice (Daily OOS)</h3>
            <div class="kpis" style="margin-top:10px">
              <div class="card"><div class="kpi-label">Short Champ bps</div><div id="ccShortChampSigned" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Short Chall bps</div><div id="ccShortChallSigned" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Short Delta bps</div><div id="ccShortDeltaSigned" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Short CI95</div><div id="ccShortCi95" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Short Significance</div><div id="ccShortSig" class="kpi-val">-</div></div>
            </div>
            <div class="table-wrap"><table id="ccShortTable"></table></div>
          </div>
        </div>

        <div id="analyticsDecisionsPanel" class="analytics-panel span-2">
          <div class="card span-2">
            <h3>Recent Decisions</h3>
            <div class="table-wrap"><table id="decisionsTable"></table></div>
          </div>
        </div>

        <div id="analyticsTradesPanel" class="analytics-panel span-2">
          <div class="card span-2">
            <h3>Closed Trades</h3>
            <div class="table-wrap"><table id="tradesTable"></table></div>
          </div>
        </div>

        <div id="analyticsEventsPanel" class="analytics-panel span-2">
          <div class="card span-2">
            <h3>Audit Events</h3>
            <div class="table-wrap"><table id="auditTable"></table></div>
          </div>
          <div class="card span-2" style="margin-top:10px">
            <h3>Notifications</h3>
            <div class="table-wrap"><table id="notificationsTable"></table></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function pad2(v) {
      return String(v).padStart(2, "0");
    }
    function fmtTs(value) {
      if (!value) return "-";
      const dt = new Date(value);
      if (Number.isNaN(dt.getTime())) return String(value);
      return `${dt.getFullYear()}-${pad2(dt.getMonth() + 1)}-${pad2(dt.getDate())} ${pad2(dt.getHours())}:${pad2(dt.getMinutes())}`;
    }
    function isTsKey(key) {
      const k = String(key || "").toLowerCase();
      return k === "ts" || k === "timestamp" || k.endsWith("_at");
    }
    function n(x, digits = 2) {
      if (x === null || x === undefined || Number.isNaN(Number(x))) return "-";
      return Number(x).toLocaleString(undefined, { maximumFractionDigits: digits });
    }
    function pct(x) {
      if (x === null || x === undefined || Number.isNaN(Number(x))) return "-";
      return (Number(x) * 100).toFixed(2) + "%";
    }
    function badge(value, cls = "") {
      return `<span class="badge ${cls}">${value ?? "-"}</span>`;
    }
    function setFlash(msg) {
      const el = document.getElementById("flash");
      el.textContent = msg;
      setTimeout(() => { if (el.textContent === msg) el.textContent = ""; }, 2200);
    }
    async function fetchJson(url, opts) {
      const r = await fetch(url, opts);
      if (!r.ok) throw new Error(url + " -> " + r.status);
      return await r.json();
    }
    async function post(url, msg) {
      try {
        await fetchJson(url, { method: "POST" });
        setFlash(msg);
        await refreshAll();
      } catch (e) {
        setFlash(String(e));
      }
    }
    async function runResearch(path, okMsg) {
      try {
        const out = await fetchJson(path, { method: "POST" });
        const p = out && out.report_path ? ` (${out.report_path})` : "";
        setFlash(okMsg + p);
        await refreshAll();
      } catch (e) {
        setFlash(String(e));
      }
    }
    async function setAccelerationMode(mode) {
      try {
        await fetchJson(`/api/acceleration/mode?mode=${encodeURIComponent(mode)}`, { method: "POST" });
        setFlash(`Acceleration mode switched to ${mode}`);
        await refreshAll();
      } catch (e) {
        setFlash(String(e));
      }
    }
    function showTab(tab) {
      const ov = document.getElementById("panelOverview");
      const an = document.getElementById("panelAnalytics");
      const b1 = document.getElementById("tabOverviewBtn");
      const b2 = document.getElementById("tabAnalyticsBtn");
      const overview = tab !== "analytics";
      if (ov) ov.className = overview ? "panel active" : "panel";
      if (an) an.className = overview ? "panel" : "panel active";
      if (b1) b1.className = overview ? "tabbtn active" : "tabbtn";
      if (b2) b2.className = overview ? "tabbtn" : "tabbtn active";
    }
    function showAnalyticsSubtab(tab) {
      const names = ["charts", "quality", "inventory", "patterns", "context", "symbolGate", "cc", "decisions", "trades", "events"];
      for (const name of names) {
        const panel = document.getElementById(`analytics${name[0].toUpperCase()}${name.slice(1)}Panel`);
        const btn = document.getElementById(`subtab${name[0].toUpperCase()}${name.slice(1)}Btn`);
        const active = name === tab;
        if (panel) panel.className = active ? "analytics-panel active span-2" : "analytics-panel span-2";
        if (btn) btn.className = active ? "subtabbtn active" : "subtabbtn";
      }
    }

    function renderKV(elId, obj, keys) {
      const el = document.getElementById(elId);
      if (!el) return;
      const list = keys || Object.keys(obj || {});
      el.innerHTML = list.map((k) => {
        const raw = obj ? obj[k] : undefined;
        let v = "-";
        if (raw !== null && raw !== undefined && raw !== "") {
          v = (typeof raw === "object")
            ? `<span class="mono">${JSON.stringify(raw)}</span>`
            : (isTsKey(k) ? fmtTs(raw) : String(raw));
        }
        return `<div class="k">${k}</div><div class="v">${v}</div>`;
      }).join("");
    }

    function renderDecisions(items) {
      const el = document.getElementById("decisionsTable");
      const rows = (items || []).map((r) => {
        const d = r.decision || {};
        const meta = d.metadata || {};
        const actionCls = d.action === "trade" ? "b-trade" : "b-hold";
        const dirCls = d.direction === "LONG" ? "b-long" : (d.direction === "SHORT" ? "b-short" : "");
        return `
          <tr>
            <td class="mono">${fmtTs(r.timestamp)}</td>
            <td>${meta.symbol || "-"}</td>
            <td>${meta.collection_role || "primary"}</td>
            <td>${badge(d.action || "-", actionCls)}</td>
            <td>${badge(d.direction || "-", dirCls)}</td>
            <td>${n(d.confidence, 3)}</td>
            <td>${n(d.size, 0)}</td>
            <td>${n(d.sl_ticks, 0)} / ${n(d.tp_ticks, 0)}</td>
            <td><span class="ellipsis" title="${(r.note || "-").replace(/"/g, "&quot;")}">${r.note || "-"}</span></td>
          </tr>
        `;
      }).join("");
      el.innerHTML = `
        <thead>
          <tr>
            <th>Timestamp</th><th>Symbol</th><th>Role</th><th>Action</th><th>Direction</th><th>Conf</th><th>Size</th><th>SL/TP</th><th>Note</th>
          </tr>
        </thead>
        <tbody>${rows || '<tr><td colspan="9" class="small">No decisions yet</td></tr>'}</tbody>
      `;
    }

    function renderTrades(items) {
      const el = document.getElementById("tradesTable");
      const rows = (items || []).map((r) => {
        const dirCls = r.direction === "LONG" ? "b-long" : (r.direction === "SHORT" ? "b-short" : "");
        return `
          <tr>
            <td class="mono">${fmtTs(r.timestamp)}</td>
            <td>${r.symbol || "-"}</td>
            <td>${badge(r.direction || "-", dirCls)}</td>
            <td>${n(r.size, 0)}</td>
            <td>${n(r.entry_price)}</td>
            <td>${n(r.exit_price)}</td>
            <td>${n(r.pnl)}</td>
            <td class="small"><span class="ellipsis" title="${(r.thesis || "-").replace(/"/g, "&quot;")}">${r.thesis || "-"}</span></td>
          </tr>
        `;
      }).join("");
      el.innerHTML = `
        <thead>
          <tr>
            <th>Timestamp</th><th>Symbol</th><th>Direction</th><th>Size</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Thesis</th>
          </tr>
        </thead>
        <tbody>${rows || '<tr><td colspan="8" class="small">No closed trades yet</td></tr>'}</tbody>
      `;
    }

    function renderAudit(items) {
      const el = document.getElementById("auditTable");
      const rows = (items || []).map((r) => {
        return `
          <tr>
            <td class="mono">${fmtTs(r.ts)}</td>
            <td>${r.event_type || "-"}</td>
            <td class="small">${JSON.stringify(r.payload || {})}</td>
            <td class="mono">${r.hash || "-"}</td>
          </tr>
        `;
      }).join("");
      el.innerHTML = `
        <thead>
          <tr>
            <th>Timestamp</th><th>Type</th><th>Payload</th><th>Hash</th>
          </tr>
        </thead>
        <tbody>${rows || '<tr><td colspan="4" class="small">No audit events yet</td></tr>'}</tbody>
      `;
    }
    function renderNotifications(items) {
      const el = document.getElementById("notificationsTable");
      const rows = (items || []).map((r) => {
        return `
          <tr>
            <td class="mono">${fmtTs(r.ts)}</td>
            <td>${r.event_type || "-"}</td>
            <td class="small">${JSON.stringify(r.payload || {})}</td>
          </tr>
        `;
      }).join("");
      el.innerHTML = `
        <thead>
          <tr>
            <th>Timestamp</th><th>Event</th><th>Payload</th>
          </tr>
        </thead>
        <tbody>${rows || '<tr><td colspan="3" class="small">No notifications yet</td></tr>'}</tbody>
      `;
    }
    function setChartEmpty(elId, text) {
      const el = document.getElementById(elId);
      if (!el) return;
      el.innerHTML = `<div class="chart-empty">${text}</div>`;
    }
    function renderEquityChart(items) {
      const trades = (items || []).slice().reverse();
      if (!trades.length) {
        setChartEmpty("eqChart", "No closed trades yet");
        return;
      }
      let eq = 0;
      const points = trades.map((t, i) => {
        eq += Number(t.pnl || 0);
        return { x: i, y: eq };
      });
      const minY = Math.min(...points.map((p) => p.y));
      const maxY = Math.max(...points.map((p) => p.y));
      const width = 520;
      const height = 180;
      const padX = 24;
      const padY = 16;
      const dx = Math.max(1, width - padX * 2);
      const dy = Math.max(1, height - padY * 2);
      const yLo = (minY === maxY) ? (minY - 1) : minY;
      const yHi = (minY === maxY) ? (maxY + 1) : maxY;
      const toX = (i) => padX + ((i / Math.max(1, points.length - 1)) * dx);
      const toY = (v) => padY + ((yHi - v) / Math.max(1e-9, (yHi - yLo))) * dy;
      const poly = points.map((p, i) => `${toX(i)},${toY(p.y)}`).join(" ");
      const baseY = toY(0);
      const svg = `
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <line x1="${padX}" y1="${baseY}" x2="${width - padX}" y2="${baseY}" stroke="#cbd5e1" stroke-width="1"/>
          <polyline points="${poly}" fill="none" stroke="#0ea5e9" stroke-width="2"/>
          <text x="${padX}" y="12" font-size="10" fill="#475569">max ${n(yHi, 2)}</text>
          <text x="${padX}" y="${height - 4}" font-size="10" fill="#475569">min ${n(yLo, 2)}</text>
        </svg>
      `;
      const el = document.getElementById("eqChart");
      if (el) el.innerHTML = svg;
    }
    function renderPnlHistogram(items) {
      const pnls = (items || []).map((t) => Number(t.pnl || 0)).filter((v) => Number.isFinite(v));
      if (!pnls.length) {
        setChartEmpty("pnlHist", "No closed trades yet");
        return;
      }
      const bins = Math.min(12, Math.max(6, Math.floor(Math.sqrt(pnls.length))));
      const minV = Math.min(...pnls);
      const maxV = Math.max(...pnls);
      const span = Math.max(1e-9, maxV - minV);
      const counts = Array.from({ length: bins }, () => 0);
      for (const v of pnls) {
        let idx = Math.floor(((v - minV) / span) * bins);
        if (idx >= bins) idx = bins - 1;
        if (idx < 0) idx = 0;
        counts[idx] += 1;
      }
      const maxC = Math.max(...counts, 1);
      const width = 520;
      const height = 180;
      const pad = 20;
      const plotW = width - pad * 2;
      const plotH = height - pad * 2;
      const barW = plotW / bins;
      const rects = counts.map((c, i) => {
        const h = (c / maxC) * plotH;
        const x = pad + (i * barW) + 1;
        const y = pad + (plotH - h);
        return `<rect x="${x}" y="${y}" width="${Math.max(1, barW - 2)}" height="${h}" fill="#14b8a6" opacity="0.85"/>`;
      }).join("");
      const svg = `
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <line x1="${pad}" y1="${pad + plotH}" x2="${width - pad}" y2="${pad + plotH}" stroke="#cbd5e1" stroke-width="1"/>
          ${rects}
          <text x="${pad}" y="12" font-size="10" fill="#475569">min ${n(minV, 2)}</text>
          <text x="${width - pad - 70}" y="12" font-size="10" fill="#475569">max ${n(maxV, 2)}</text>
        </svg>
      `;
      const el = document.getElementById("pnlHist");
      if (el) el.innerHTML = svg;
    }
    function renderDecisionMix(items) {
      const rows = items || [];
      if (!rows.length) {
        setChartEmpty("decMix", "No decisions yet");
        return;
      }
      let trade = 0;
      let hold = 0;
      let longDir = 0;
      let shortDir = 0;
      for (const r of rows) {
        const d = r.decision || {};
        const action = String(d.action || "").toLowerCase();
        const dir = String(d.direction || "").toUpperCase();
        if (action === "trade") trade += 1;
        if (action === "hold") hold += 1;
        if (dir === "LONG") longDir += 1;
        if (dir === "SHORT") shortDir += 1;
      }
      const total = Math.max(1, trade + hold);
      const width = 520;
      const height = 180;
      const barW = 70;
      const gap = 38;
      const x0 = 80;
      const baseline = 140;
      const scale = 90 / total;
      const bars = [
        { label: "Trade", value: trade, color: "#0ea5e9" },
        { label: "Hold", value: hold, color: "#f59e0b" },
        { label: "Long", value: longDir, color: "#16a34a" },
        { label: "Short", value: shortDir, color: "#dc2626" },
      ];
      const rects = bars.map((b, i) => {
        const h = Math.max(2, b.value * scale);
        const x = x0 + i * (barW + gap);
        const y = baseline - h;
        return `
          <rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="${b.color}" opacity="0.88"/>
          <text x="${x + (barW / 2)}" y="${baseline + 14}" text-anchor="middle" font-size="10" fill="#475569">${b.label}</text>
          <text x="${x + (barW / 2)}" y="${y - 4}" text-anchor="middle" font-size="10" fill="#334155">${b.value}</text>
        `;
      }).join("");
      const svg = `
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <line x1="50" y1="${baseline}" x2="${width - 30}" y2="${baseline}" stroke="#cbd5e1" stroke-width="1"/>
          ${rects}
          <text x="16" y="12" font-size="10" fill="#475569">n=${rows.length}</text>
        </svg>
      `;
      const el = document.getElementById("decMix");
      if (el) el.innerHTML = svg;
    }
    function metricRows(obj) {
      return Object.entries(obj || {})
        .map(([name, m]) => ({ name, ...(m || {}) }))
        .filter((r) => Number(r.count || 0) > 0)
        .sort((a, b) => Number(b.count || 0) - Number(a.count || 0));
    }
    function renderQualityTable(elId, rows) {
      const el = document.getElementById(elId);
      if (!el) return;
      const body = (rows || []).map((r) => `
        <tr>
          <td>${r.name}</td>
          <td>${n(r.count, 0)}</td>
          <td>${pct(r.accuracy)}</td>
          <td>${n(r.avg_signed_return_bps, 2)}</td>
          <td>${n(r.brier_score, 3)}</td>
        </tr>
      `).join("");
      el.innerHTML = `
        <thead><tr><th>Group</th><th>N</th><th>Accuracy</th><th>Signed bps</th><th>Brier</th></tr></thead>
        <tbody>${body || '<tr><td colspan="5" class="small">Need more labelled predictions</td></tr>'}</tbody>
      `;
    }
    function renderQualityChart(horizons) {
      const entries = Object.values(horizons || {})
        .map((h) => ({
          horizon: Number(h.horizon_minutes || 0),
          count: Number((h.overall || {}).count || 0),
          signed: Number((h.overall || {}).avg_signed_return_bps || 0),
          accuracy: Number((h.overall || {}).accuracy || 0),
        }))
        .filter((h) => h.horizon > 0)
        .sort((a, b) => a.horizon - b.horizon);
      if (!entries.length || entries.every((h) => h.count <= 0)) {
        setChartEmpty("qualityChart", "Collect more post-upgrade samples");
        return;
      }
      const width = 520;
      const height = 180;
      const baseline = 92;
      const maxAbs = Math.max(1, ...entries.map((h) => Math.abs(h.signed)));
      const barW = Math.max(32, Math.min(76, 320 / Math.max(1, entries.length)));
      const gap = 28;
      const x0 = Math.max(34, (width - ((barW + gap) * entries.length)) / 2);
      const bars = entries.map((h, i) => {
        const mag = Math.abs(h.signed) / maxAbs * 62;
        const x = x0 + i * (barW + gap);
        const y = h.signed >= 0 ? baseline - mag : baseline;
        const color = h.signed >= 0 ? "#16a34a" : "#dc2626";
        return `
          <rect x="${x}" y="${y}" width="${barW}" height="${Math.max(2, mag)}" fill="${color}" opacity="0.88"/>
          <text x="${x + (barW / 2)}" y="164" text-anchor="middle" font-size="10" fill="#475569">${h.horizon}m</text>
          <text x="${x + (barW / 2)}" y="${h.signed >= 0 ? y - 5 : y + mag + 12}" text-anchor="middle" font-size="10" fill="#334155">${n(h.signed, 1)}</text>
          <text x="${x + (barW / 2)}" y="176" text-anchor="middle" font-size="9" fill="#64748b">n=${h.count} ${pct(h.accuracy)}</text>
        `;
      }).join("");
      const el = document.getElementById("qualityChart");
      if (el) {
        el.innerHTML = `
          <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
            <line x1="28" y1="${baseline}" x2="${width - 28}" y2="${baseline}" stroke="#cbd5e1" stroke-width="1"/>
            ${bars}
          </svg>
        `;
      }
    }
    function renderPredictionQuality(report, controls) {
      const q = (report || {}).prediction_quality || {};
      const horizons = q.horizons || {};
      const h15 = horizons["15"] || horizons["5"] || Object.values(horizons)[0] || {};
      const overall = h15.overall || {};
      document.getElementById("qLabels").textContent = n(overall.count, 0);
      document.getElementById("qAccuracy").textContent = pct(overall.accuracy);
      document.getElementById("qSigned").textContent = `${n(overall.avg_signed_return_bps, 2)} bps`;
      document.getElementById("qBrier").textContent = n(overall.brier_score, 3);
      document.getElementById("qQuote").textContent = `${n(q.samples_with_quote, 0)} / ${n(q.filtered_input_sample_count || q.input_sample_count, 0)}`;
      renderQualityChart(horizons);
      renderQualityTable("qualitySymbolTable", metricRows(h15.by_symbol));
      renderQualityTable("qualityConfidenceTable", metricRows(h15.by_confidence_bin));
      const note = document.getElementById("qualityNote");
      if (note) {
        note.textContent = `Mode: ${q.quality_mode || "all"}. Using ${h15.horizon_minutes || "-"}m horizon. Eligible directional predictions: ${n((q.eligible_directional_predictions ?? q.eligible_trade_predictions), 0)}. Old samples without quote labels skipped: ${n(q.skipped_no_quote, 0)}.`;
      }
      const c = (controls || {}).quality_controls || {};
      const cc = c.confidence_controls || {};
      document.getElementById("qAllowedCells").textContent = n(c.regime_allowed_cells, 0);
      document.getElementById("qBestHorizonSymbols").textContent = n(c.best_horizon_symbols, 0);
      document.getElementById("qSampleSkipRate").textContent = pct(c.sample_balance_skip_rate);
      const g = c.coverage_guard || {};
      document.getElementById("qCoverageGuard").textContent = g.active
        ? `Active (${n(g.daily_labels, 0)}/${n(g.target, 0)})`
        : `Off (${n(g.daily_labels, 0)}/${n(g.target, 0)})`;
      const pt = c.policy_tier || {};
      const tier = String(pt.active || "-").toLowerCase();
      const tierText = tier ? tier.charAt(0).toUpperCase() + tier.slice(1) : "-";
      document.getElementById("qPolicyTier").textContent = tierText;
      const sq = c.symbol_quarantine || {};
      const sqActive = sq.active || {};
      const sqSymbols = Object.keys(sqActive || {});
      document.getElementById("qSymbolQuarantine").textContent = `${n(sq.count,0)} active`;
      const ccControls = cc.controls || {};
      const calEnabled = Boolean(ccControls.calibration_gate_enabled);
      document.getElementById("qCalGate").textContent = calEnabled
        ? `ON (stress ${n(ccControls.calibration_gate_stress_bps, 2)} bps)`
        : "OFF";
      document.getElementById("qCalEval").textContent = fmtTs(cc.evaluated_at);
      renderCalibrationGateTable(cc);
      renderDirectionPolicyTable(cc);
      const perf = ((pt.performance_15m || {}).rows || []);
      const activeRow = perf.find((r) => String(r.policy_tier || "").toLowerCase() === tier);
      const perfText = activeRow
        ? `Active tier=${tierText} | n=${n(activeRow.count,0)} | signed=${n(activeRow.avg_signed_return_bps,2)} bps | acc=${pct(activeRow.accuracy)}`
        : `Active tier=${tierText}`;
      const quarantineText = sqSymbols.length
        ? `Quarantine: ${sqSymbols.join(", ")}`
        : "Quarantine: none";
      if (note) {
        note.textContent = `Mode: ${q.quality_mode || "all"}. Using ${h15.horizon_minutes || "-"}m horizon. Eligible directional predictions: ${n((q.eligible_directional_predictions ?? q.eligible_trade_predictions), 0)}. Old samples without quote labels skipped: ${n(q.skipped_no_quote, 0)}. ${perfText}. ${quarantineText}.`;
      }
    }
    function renderCalibrationGateTable(confidenceControls) {
      const el = document.getElementById("qualityCalibTable");
      if (!el) return;
      const summaryEl = document.getElementById("qCalSummary");
      const controls = (confidenceControls || {}).controls || {};
      const allowed = controls.symbol_allowed_confidence_bins || {};
      const stressed = controls.symbol_bin_stressed_bps || {};
      const symbols = Array.from(new Set([].concat(Object.keys(allowed || {}), Object.keys(stressed || {})))).sort();
      const records = [];
      for (const sym of symbols) {
        const allowedSet = new Set((allowed[sym] || []).map((x) => String(x)));
        const sMap = stressed[sym] || {};
        const bins = Object.keys(sMap).sort();
        for (const bin of bins) {
          const sbps = Number(sMap[bin] || 0);
          const ok = allowedSet.has(String(bin));
          records.push({ sym, bin, sbps, ok });
        }
      }
      records.sort((a, b) => {
        if (a.sbps !== b.sbps) return a.sbps - b.sbps;
        if (a.sym !== b.sym) return a.sym.localeCompare(b.sym);
        return a.bin.localeCompare(b.bin);
      });
      const blockedCount = records.filter((r) => !r.ok).length;
      const allowedCount = records.filter((r) => r.ok).length;
      const worst = records.length ? records[0] : null;
      if (summaryEl) {
        summaryEl.textContent = worst
          ? `Blocked bins: ${n(blockedCount, 0)} | Allowed bins: ${n(allowedCount, 0)} | Worst: ${worst.sym} ${worst.bin} (${n(worst.sbps, 2)} bps)`
          : "No calibration bin history yet";
      }
      const rows = records.map((r) => `
            <tr>
              <td>${r.sym}</td>
              <td>${r.bin}</td>
              <td>${n(r.sbps, 2)}</td>
              <td>${badge(r.ok ? "allowed" : "blocked", r.ok ? "b-long" : "b-short")}</td>
            </tr>
          `).join("");
      el.innerHTML = `
        <thead><tr><th>Symbol</th><th>Bin</th><th>Stressed Signed Bps</th><th>Status</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4" class="small">No calibration bin history yet</td></tr>'}</tbody>
      `;
    }
    function renderDirectionPolicyTable(confidenceControls) {
      const el = document.getElementById("qualityDirectionPolicyTable");
      if (!el) return;
      const summaryEl = document.getElementById("qDirectionSummary");
      const controls = (confidenceControls || {}).controls || {};
      const policy = controls.symbol_direction_policy || {};
      const blocked = controls.symbol_blocked_directions || {};
      const records = [];
      for (const sym of Object.keys(policy || {}).sort()) {
        const byDir = policy[sym] || {};
        for (const dir of Object.keys(byDir).sort()) {
          const row = byDir[dir] || {};
          records.push({
            sym,
            dir,
            status: String(row.status || "-"),
            count: Number(row.count || 0),
            accuracy: Number(row.accuracy || 0),
            signed: Number(row.avg_signed_return_bps || 0),
            stressed: Number(row.stressed_signed_bps || 0),
          });
        }
      }
      const blockedLabels = [];
      for (const sym of Object.keys(blocked || {}).sort()) {
        for (const dir of (blocked[sym] || [])) blockedLabels.push(`${sym} ${dir}`);
      }
      if (summaryEl) {
        summaryEl.textContent = blockedLabels.length
          ? `Blocked: ${blockedLabels.join(", ")}`
          : "No blocked directions";
      }
      const rows = records.map((r) => `
            <tr>
              <td>${r.sym}</td>
              <td>${r.dir}</td>
              <td>${n(r.count, 0)}</td>
              <td>${pct(r.accuracy)}</td>
              <td>${n(r.signed, 2)}</td>
              <td>${n(r.stressed, 2)}</td>
              <td>${badge(r.status, r.status === "allow" ? "b-long" : (r.status === "block" ? "b-short" : "b-hold"))}</td>
            </tr>
          `).join("");
      el.innerHTML = `
        <thead><tr><th>Symbol</th><th>Direction</th><th>N</th><th>Accuracy</th><th>Signed Bps</th><th>Stressed Bps</th><th>Status</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="7" class="small">No direction policy history yet</td></tr>'}</tbody>
      `;
    }
    function renderPatternLeaderboard(report) {
      const pl = (report || {}).pattern_leaderboard || {};
      document.getElementById("plAllowedCount").textContent = n(pl.allowed_count, 0);
      document.getElementById("plStressBps").textContent = `${n(pl.stress_bps, 2)} bps`;
      document.getElementById("plMinAccuracy").textContent = pct(pl.min_accuracy);
      document.getElementById("plMinLabels").textContent = n(pl.min_labels, 0);
      const rows = pl.top || pl.rows || [];
      const summaryEl = document.getElementById("plSummary");
      const topAllowed = (pl.allowed || [])[0];
      if (summaryEl) {
        summaryEl.textContent = topAllowed
          ? `Best allowed: ${topAllowed.symbol} ${topAllowed.direction} ${n(topAllowed.horizon_minutes, 0)}m | ${topAllowed.feature}=${topAllowed.value} | stressed=${n(topAllowed.stressed_signed_bps, 2)} bps | acc=${pct(topAllowed.accuracy)} | n=${n(topAllowed.count, 0)}`
          : "No allowed pattern states yet. Older samples may still be missing pattern metadata.";
      }
      const el = document.getElementById("patternLeaderboardTable");
      if (!el) return;
      const body = rows.map((r) => `
        <tr>
          <td>${r.symbol || "-"}</td>
          <td>${r.direction || "-"}</td>
          <td>${n(r.horizon_minutes, 0)}m</td>
          <td>${r.feature || "-"}</td>
          <td>${r.value || "-"}</td>
          <td>${n(r.count, 0)}</td>
          <td>${pct(r.accuracy)}</td>
          <td>${n(r.avg_signed_return_bps, 2)}</td>
          <td>${n(r.stressed_signed_bps, 2)}</td>
          <td>${badge(r.status || "-", r.status === "allow" ? "b-long" : (r.status === "block" ? "b-short" : "b-hold"))}</td>
        </tr>
      `).join("");
      el.innerHTML = `
        <thead><tr><th>Symbol</th><th>Dir</th><th>Horizon</th><th>Feature</th><th>Value</th><th>N</th><th>Accuracy</th><th>Signed bps</th><th>Stressed bps</th><th>Status</th></tr></thead>
        <tbody>${body || '<tr><td colspan="10" class="small">No pattern leaderboard rows yet</td></tr>'}</tbody>
      `;
    }
    function renderFeatureInventory(report) {
      const fi = (report || {}).feature_inventory || {};
      document.getElementById("fiFeatureCount").textContent = n(fi.feature_count, 0);
      document.getElementById("fiSampleCount").textContent = n(fi.sample_count, 0);
      const byCat = fi.by_category || {};
      document.getElementById("fiCoreCount").textContent = n(byCat.core, 0);
      document.getElementById("fiPatternCount").textContent = n(byCat.pattern, 0);
      document.getElementById("fiStructureCount").textContent = n(byCat.structure, 0);
      const items = fi.items || [];
      const fullyCovered = items.filter((x) => Number(x.coverage_pct || 0) >= 95).length;
      const summaryEl = document.getElementById("fiSummary");
      if (summaryEl) {
        summaryEl.textContent = `${n(fullyCovered, 0)} / ${n(items.length, 0)} features have at least 95% populated coverage. Unknown-heavy fields usually indicate older pre-feature samples.`;
      }
      const el = document.getElementById("featureInventoryTable");
      if (!el) return;
      const body = items.map((r) => `
        <tr>
          <td>${r.category || "-"}</td>
          <td>${r.label || r.feature || "-"}</td>
          <td>${r.feature || "-"}</td>
          <td>${r.kind || "-"}</td>
          <td>${n(r.present_count, 0)}</td>
          <td>${n(r.unknown_count, 0)}</td>
          <td>${n(r.coverage_pct, 1)}%</td>
          <td>${(r.examples || []).join(", ") || "-"}</td>
        </tr>
      `).join("");
      el.innerHTML = `
        <thead><tr><th>Category</th><th>Label</th><th>Feature</th><th>Type</th><th>Present</th><th>Unknown</th><th>Coverage</th><th>Examples</th></tr></thead>
        <tbody>${body || '<tr><td colspan="8" class="small">No feature inventory yet</td></tr>'}</tbody>
      `;
    }
    function renderContextLeaderboard(report) {
      const cl = (report || {}).context_leaderboard || {};
      document.getElementById("clAllowedCount").textContent = n(cl.allowed_count, 0);
      document.getElementById("clStressBps").textContent = `${n(cl.stress_bps, 2)} bps`;
      document.getElementById("clMinAccuracy").textContent = pct(cl.min_accuracy);
      document.getElementById("clMinLabels").textContent = n(cl.min_labels, 0);
      const rows = cl.top || cl.rows || [];
      const summaryEl = document.getElementById("clSummary");
      const topAllowed = (cl.allowed || [])[0];
      if (summaryEl) {
        summaryEl.textContent = topAllowed
          ? `Best allowed: ${topAllowed.symbol} ${topAllowed.direction} ${n(topAllowed.horizon_minutes, 0)}m | ${topAllowed.feature}=${topAllowed.value} | stressed=${n(topAllowed.stressed_signed_bps, 2)} bps | acc=${pct(topAllowed.accuracy)} | n=${n(topAllowed.count, 0)}`
          : "No allowed context states yet. Fresh labelled samples are needed for the new structure fields.";
      }
      const el = document.getElementById("contextLeaderboardTable");
      if (!el) return;
      const body = rows.map((r) => `
        <tr>
          <td>${r.symbol || "-"}</td>
          <td>${r.direction || "-"}</td>
          <td>${n(r.horizon_minutes, 0)}m</td>
          <td>${r.feature || "-"}</td>
          <td>${r.value || "-"}</td>
          <td>${n(r.count, 0)}</td>
          <td>${pct(r.accuracy)}</td>
          <td>${n(r.avg_signed_return_bps, 2)}</td>
          <td>${n(r.stressed_signed_bps, 2)}</td>
          <td>${badge(r.status || "-", r.status === "allow" ? "b-long" : (r.status === "block" ? "b-short" : "b-hold"))}</td>
        </tr>
      `).join("");
      el.innerHTML = `
        <thead><tr><th>Symbol</th><th>Dir</th><th>Horizon</th><th>Feature</th><th>Value</th><th>N</th><th>Accuracy</th><th>Signed bps</th><th>Stressed bps</th><th>Status</th></tr></thead>
        <tbody>${body || '<tr><td colspan="10" class="small">No context leaderboard rows yet</td></tr>'}</tbody>
      `;
    }
    function renderSymbolGate(report) {
      const sg = (report || {}).symbol_performance || {};
      const rows = sg.rows || [];
      const allowed = sg.allowed_symbols || [];
      const blocked = sg.blocked_symbols || [];
      document.getElementById("sgAllowed").textContent = n(allowed.length, 0);
      document.getElementById("sgBlocked").textContent = n(blocked.length, 0);
      document.getElementById("sgHorizon").textContent = `${n(sg.horizon_minutes, 0)}m`;
      const el = document.getElementById("symbolGateTable");
      if (!el) return;
      const body = rows.map((r) => `
        <tr>
          <td>${r.symbol || "-"}</td>
          <td>${n(r.count, 0)}</td>
          <td>${pct(r.accuracy)}</td>
          <td>${n(r.avg_signed_return_bps, 2)}</td>
          <td>${n(r.brier_score, 3)}</td>
          <td>${badge(r.recommendation || "-", r.recommendation === "allow" ? "b-long" : (r.recommendation === "block" ? "b-short" : "b-hold"))}</td>
        </tr>
      `).join("");
      el.innerHTML = `
        <thead><tr><th>Symbol</th><th>N</th><th>Accuracy</th><th>Signed Bps</th><th>Brier</th><th>Recommendation</th></tr></thead>
        <tbody>${body || '<tr><td colspan="6" class="small">No symbol stats yet</td></tr>'}</tbody>
      `;
    }
    function renderCellBootstrap(report) {
      const cb = (report || {}).cell_leaderboard_bootstrap || {};
      document.getElementById("sgRobustCells").textContent = n(cb.robust_count, 0);
      const rows = cb.top || cb.rows || [];
      const el = document.getElementById("cellBootstrapTable");
      if (!el) return;
      const body = rows.map((r) => `
        <tr>
          <td>${r.symbol || "-"}</td>
          <td>${r.regime || "-"}</td>
          <td>${r.session_bucket || "-"}</td>
          <td>${n(r.horizon_minutes,0)}m</td>
          <td>${n(r.count,0)}</td>
          <td>${n(r.avg_signed_return_bps,2)}</td>
          <td>[${n(r.ci95_low_signed_bps,2)}, ${n(r.ci95_high_signed_bps,2)}]</td>
          <td>${badge(r.robust_positive ? "yes" : "no", r.robust_positive ? "b-long" : "b-hold")}</td>
        </tr>
      `).join("");
      el.innerHTML = `
        <thead><tr><th>Symbol</th><th>Regime</th><th>Session</th><th>Horizon</th><th>N</th><th>Signed bps</th><th>CI95</th><th>Robust</th></tr></thead>
        <tbody>${body || '<tr><td colspan="8" class="small">No robust cells yet</td></tr>'}</tbody>
      `;
    }
    function renderChampionChallenger(report, shortReport) {
      const cc = (report || {}).champion_challenger_daily || {};
      const c = cc.champion_overall || {};
      const h = cc.challenger_overall || {};
      const s = cc.significance || {};
      document.getElementById("ccChampSigned").textContent = n(c.signed_bps, 2);
      document.getElementById("ccChallSigned").textContent = n(h.signed_bps, 2);
      document.getElementById("ccDeltaSigned").textContent = n(cc.overall_delta_signed_bps, 2);
      document.getElementById("ccCi95").textContent = `[${n(s.ci95_low,2)}, ${n(s.ci95_high,2)}]`;
      document.getElementById("ccSig").textContent = s.significant_positive ? "Positive (95%)" : "Not significant";
      const rows = cc.days || [];
      const el = document.getElementById("ccTable");
      if (!el) return;
      const body = rows.map((r) => `
        <tr>
          <td>${r.day || "-"}</td>
          <td>${n(r.champion_count,0)}</td>
          <td>${n(r.champion_signed_bps,2)}</td>
          <td>${n(r.challenger_count,0)}</td>
          <td>${n(r.challenger_signed_bps,2)}</td>
          <td>${n(r.delta_signed_bps,2)}</td>
          <td>${n(r.allowed_cells,0)}</td>
        </tr>
      `).join("");
      el.innerHTML = `
        <thead><tr><th>Day</th><th>Champ N</th><th>Champ bps</th><th>Chall N</th><th>Chall bps</th><th>Delta bps</th><th>Allowed Cells</th></tr></thead>
        <tbody>${body || '<tr><td colspan="7" class="small">No OOS days yet</td></tr>'}</tbody>
      `;

      const ccShort = (shortReport || {}).champion_challenger_daily || {};
      const sc = ccShort.champion_overall || {};
      const sh = ccShort.challenger_overall || {};
      const ss = ccShort.significance || {};
      document.getElementById("ccShortChampSigned").textContent = n(sc.signed_bps, 2);
      document.getElementById("ccShortChallSigned").textContent = n(sh.signed_bps, 2);
      document.getElementById("ccShortDeltaSigned").textContent = n(ccShort.overall_delta_signed_bps, 2);
      document.getElementById("ccShortCi95").textContent = `[${n(ss.ci95_low,2)}, ${n(ss.ci95_high,2)}]`;
      document.getElementById("ccShortSig").textContent = ss.significant_positive ? "Positive (95%)" : "Not significant";
      const sRows = ccShort.days || [];
      const sEl = document.getElementById("ccShortTable");
      if (!sEl) return;
      const sBody = sRows.map((r) => `
        <tr>
          <td>${r.day || "-"}</td>
          <td>${n(r.champion_count,0)}</td>
          <td>${n(r.champion_signed_bps,2)}</td>
          <td>${n(r.challenger_count,0)}</td>
          <td>${n(r.challenger_signed_bps,2)}</td>
          <td>${n(r.delta_signed_bps,2)}</td>
          <td>${n(r.allowed_cells,0)}</td>
        </tr>
      `).join("");
      sEl.innerHTML = `
        <thead><tr><th>Day</th><th>Champ N</th><th>Champ bps</th><th>Chall N</th><th>Chall bps</th><th>Delta bps</th><th>Allowed Cells</th></tr></thead>
        <tbody>${sBody || '<tr><td colspan="7" class="small">No short-slice OOS days yet</td></tr>'}</tbody>
      `;
    }

    function setHeaderChips(status, gate) {
      const cWorker = document.getElementById("chipWorker");
      const cGate = document.getElementById("chipGoLive");
      const cMode = document.getElementById("chipMode");

      cWorker.className = "chip " + (status.running ? "ok" : "warn");
      cWorker.textContent = status.running ? "Worker: running" : "Worker: stopped";

      if (gate.passed) {
        cGate.className = "chip ok";
        cGate.textContent = "Go-Live: passed";
      } else {
        cGate.className = "chip warn";
        cGate.textContent = "Go-Live: not passed";
      }

      const mode = gate.live_mode ? "Live" : "Paper/Mock";
      cMode.className = gate.live_mode ? "chip danger" : "chip";
      cMode.textContent = "Mode: " + mode;
    }
    function parseHHMM(value) {
      const raw = String(value || "");
      const p = raw.split(":");
      if (p.length !== 2) return [0, 0];
      const hh = Math.max(0, Math.min(23, Number(p[0]) || 0));
      const mm = Math.max(0, Math.min(59, Number(p[1]) || 0));
      return [hh, mm];
    }
    function nyNowHM(tzName) {
      const parts = new Intl.DateTimeFormat("en-US", {
        timeZone: tzName || "America/New_York",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
      }).formatToParts(new Date());
      const hh = Number((parts.find((x) => x.type === "hour") || {}).value || "0");
      const mm = Number((parts.find((x) => x.type === "minute") || {}).value || "0");
      return [hh, mm];
    }
    function fmtHM(h, m) {
      return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
    }
    function fmtDelta(mins) {
      const total = Math.max(0, Math.floor(mins));
      const h = Math.floor(total / 60);
      const m = total % 60;
      return h > 0 ? `${h}h ${m}m` : `${m}m`;
    }
    function sessionStatusText(sessionCfg) {
      if (!sessionCfg || !sessionCfg.enable_session_filter) return "Filter off";
      const [sh, sm] = parseHHMM(sessionCfg.session_start_et);
      const [eh, em] = parseHHMM(sessionCfg.session_end_et);
      const tzName = String(sessionCfg.timezone || "America/New_York");
      const tzLabel = tzName === "America/New_York" ? "NY" : tzName;
      const [nh, nm] = nyNowHM(tzName);
      const start = (sh * 60) + sm;
      const end = (eh * 60) + em;
      const now = (nh * 60) + nm;
      const wraps = start > end;
      let inSession = false;
      if (!wraps) inSession = now >= start && now <= end;
      else inSession = now >= start || now <= end;

      if (inSession) {
        const minsLeft = !wraps
          ? (end - now)
          : (now <= end ? (end - now) : ((24 * 60 - now) + end));
        return `In session (${fmtHM(nh, nm)} ${tzLabel}, closes in ${fmtDelta(minsLeft)})`;
      }
      const minsToOpen = !wraps
        ? (now < start ? (start - now) : ((24 * 60 - now) + start))
        : (now > end && now < start ? (start - now) : 0);
      if (minsToOpen > 0) return `Out of session (${fmtHM(nh, nm)} ${tzLabel}, opens in ${fmtDelta(minsToOpen)})`;
      return `Out of session (${fmtHM(nh, nm)} ${tzLabel})`;
    }
    function esc(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }
    function blockerSummary(decisions) {
      if (!decisions || decisions.length === 0) {
        return { label: "No history", detail: "No decision has been recorded yet." };
      }
      const latest = decisions[0];
      const decision = latest.decision || {};
      const reason = String(decision.reasoning || latest.note || "").trim();
      const action = String(decision.action || "").toLowerCase();
      if (!reason) {
        return { label: "No blocker", detail: action === "trade" ? "Latest decision allowed a trade." : "Latest decision did not include a reason." };
      }
      if (reason.startsWith("Session gate:")) {
        return { label: "Session window", detail: reason.replace(/^Session gate:\\s*/, "") };
      }
      if (reason.startsWith("Weekend block active;")) {
        return { label: "Weekend block", detail: "Worker is intentionally paused on Saturday and Sunday." };
      }
      if (reason.startsWith("Trading kill switch enabled:")) {
        return { label: "Kill switch enabled", detail: reason.replace(/^Trading kill switch enabled:\\s*/, "") };
      }
      if (reason.startsWith("Data quality guard active:")) {
        return { label: "Data quality guard", detail: reason.replace(/^Data quality guard active:\\s*/, "") };
      }
      if (reason.startsWith("Cost guard active:")) {
        return { label: "Cost guard", detail: reason.replace(/^Cost guard active:\\s*/, "") };
      }
      if (reason.startsWith("Signed-bps kill switch active:")) {
        return { label: "Performance kill switch", detail: reason.replace(/^Signed-bps kill switch active:\\s*/, "") };
      }
      if (reason.startsWith("Broker sync stale while position open")) {
        return { label: "Broker sync stale", detail: reason };
      }
      if (reason.startsWith("Symbol gate active:")) {
        return { label: "Symbol gate", detail: reason.replace(/^Symbol gate active:\\s*/, "") };
      }
      if (reason.startsWith("Model monitoring safe mode active:")) {
        return { label: "Model monitoring safe mode", detail: reason.replace(/^Model monitoring safe mode active:\\s*/, "") };
      }
      if (reason.startsWith("Risk override: max trades reached.")) {
        return { label: "Daily trade cap", detail: "Maximum trades for the day has been reached." };
      }
      if (reason.startsWith("Risk override: existing open position.")) {
        return { label: "Open position", detail: "Apex will not stack a new position while one is already open." };
      }
      if (reason.startsWith("Risk override: invalid direction.")) {
        return { label: "Invalid direction", detail: "The model output did not provide a valid LONG or SHORT direction." };
      }
      if (reason.startsWith("Spread gate:")) {
        return { label: "Spread too wide", detail: reason.replace(/^Spread gate:\\s*/, "") };
      }
      if (reason.startsWith("Confluence gate:")) {
        return { label: "Confluence too low", detail: reason.replace(/^Confluence gate:\\s*/, "") };
      }
      if (reason.startsWith("EV gate:")) {
        return { label: "Expected value too low", detail: reason.replace(/^EV gate:\\s*/, "") };
      }
      if (reason.startsWith("Economic calendar gate:")) {
        return { label: "Event risk", detail: reason.replace(/^Economic calendar gate:\\s*/, "") };
      }
      if (reason.startsWith("Pre-trade risk gateway blocked order:")) {
        return { label: "Pre-trade risk", detail: reason.replace(/^Pre-trade risk gateway blocked order:\\s*/, "") };
      }
      if (reason.startsWith("The market shows")) {
        return { label: "Model chose hold", detail: reason };
      }
      if (reason.startsWith("Market ")) {
        return { label: "Model chose hold", detail: reason };
      }
      return { label: action === "hold" ? "Model chose hold" : "Latest decision", detail: reason };
    }
    function summarizeBlocker(decisions) {
      const summary = blockerSummary(decisions);
      const detail = summary.detail.length > 150 ? summary.detail.slice(0, 150) + "..." : summary.detail;
      return `<span class="blocker-label">${esc(summary.label)}</span><span class="blocker-detail">${esc(detail)}</span>`;
    }

    async function refreshAll() {
      try {
        const analyticsActive = analyticsTabActive();
        const activeAnalyticsSubtab = currentAnalyticsSubtab();
        const needWideHistory = analyticsActive && activeAnalyticsSubtab === "charts";
        const fastFirstLoad = !analyticsActive && !window.__apexHydrated;
        const [
          status,
          account,
          cfgRes,
          metricsRes,
          decRes,
          trRes,
          decWideRes,
          trWideRes,
          gateRes,
          auditRes,
          notifRes,
          sessionRes,
          accelRes,
          qualityRes,
          monitorRes,
          symbolPerfRes,
          controlsRes,
          ccRes,
          ccShortRes,
          cbRes,
          autoHealthRes,
          guardsRes,
          causesRes,
          automationRes,
          readinessRes,
          inventoryRes,
          patternRes,
          contextRes
        ] = await Promise.all([
          fetchJson("/api/status"),
          fetchJson("/api/account"),
          fetchJson("/api/config"),
          fetchJson("/api/metrics"),
          fetchJson("/api/decisions?limit=15"),
          fetchJson("/api/trades?limit=15"),
          needWideHistory ? fetchJson("/api/decisions?limit=80") : Promise.resolve({ items: [] }),
          needWideHistory ? fetchJson("/api/trades?limit=80") : Promise.resolve({ items: [] }),
          fetchJson("/api/go-live-gate"),
          fastFirstLoad ? Promise.resolve({ items: [] }) : fetchJson("/api/audit?limit=20"),
          fastFirstLoad ? Promise.resolve({ items: [] }) : fetchJson("/api/notifications?limit=20"),
          fetchJson("/api/session-config"),
          fastFirstLoad ? Promise.resolve({}) : fetchJson("/api/acceleration"),
          analyticsActive && activeAnalyticsSubtab === "quality"
            ? fetchJson("/api/research/prediction-quality?lookback=600&horizons=5,15,30&min_confidence=0&quality_mode=good_only")
            : Promise.resolve({}),
          fastFirstLoad ? Promise.resolve({}) : fetchJson("/api/model-monitoring"),
          analyticsActive && activeAnalyticsSubtab === "symbolGate"
            ? fetchJson("/api/research/symbol-performance?lookback=20000&horizon_minutes=15&quality_mode=good_only")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "quality"
            ? fetchJson("/api/quality-controls")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "cc"
            ? fetchJson("/api/research/champion-challenger-daily?lookback=20000&horizon_minutes=15&quality_mode=good_only&min_train_labels=150&min_cell_labels=12&challenger_min_confidence=0.55&min_daily_selections=10")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "cc"
            ? fetchJson("/api/research/champion-challenger-daily?lookback=20000&horizon_minutes=15&quality_mode=good_only&min_train_labels=150&min_cell_labels=12&direction=SHORT&challenger_min_confidence=0.60&challenger_max_confidence=0.80&min_daily_selections=8")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "symbolGate"
            ? fetchJson("/api/research/cell-leaderboard-bootstrap?lookback=20000&horizons=5,15,30&quality_mode=good_only&min_labels=12&n_bootstrap=120&robust_only=false")
            : Promise.resolve({}),
          fastFirstLoad ? Promise.resolve({}) : fetchJson("/api/autonomy/health-score"),
          fastFirstLoad ? Promise.resolve({}) : fetchJson("/api/automation/guards"),
          fastFirstLoad ? Promise.resolve({}) : fetchJson("/api/autonomy/error-causes?window_minutes=1440"),
          fastFirstLoad ? Promise.resolve({}) : fetchJson("/api/automation"),
          fetchJson("/api/data-readiness?lookback=2000"),
          analyticsActive && activeAnalyticsSubtab === "inventory"
            ? fetchJson("/api/research/feature-inventory?lookback=2000")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "patterns"
            ? fetchJson("/api/research/pattern-leaderboard?lookback=20000&horizons=15,30&quality_mode=good_only&min_labels=5&stress_bps=3&min_accuracy=0.5")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "context"
            ? fetchJson("/api/research/context-leaderboard?lookback=20000&horizons=15,30&quality_mode=good_only&min_labels=5&stress_bps=3&min_accuracy=0.5")
            : Promise.resolve({})
        ]);

        const cfg = cfgRes.config || {};
        const metrics = metricsRes.metrics || {};
        const decisions = decRes.items || [];
        const trades = trRes.items || [];
        const decisionsWide = decWideRes.items || [];
        const tradesWide = trWideRes.items || [];
        const gate = gateRes.go_live_gate || {};
        const audits = auditRes.items || [];
        const notifications = notifRes.items || [];
        const sessionCfg = sessionRes.session || {};
        const accel = accelRes.acceleration || {};
        const monitor = monitorRes.model_monitoring || {};
        const autonomyHealth = autoHealthRes.autonomy_health || {};
        const guards = guardsRes.guards || {};
        const qGuard = guards.quality_guard || {};
        const sfGuard = guards.sample_flow_guard || {};
        const qMetrics = qGuard.metrics || {};
        const causes = causesRes.error_causes || {};
        const topCause = causes.top_cause || {};
        const llmHealth = status.llm_provider_health || {};
        const automation = automationRes.automation || {};
        const readiness = readinessRes.data_readiness || {};
        const automationState = automation.state || {};
        const weekly = automation.weekly_experiments || {};
        const weeklyLast = weekly.last || {};
        const weeklySel = (weeklyLast.selected || {});
        const dailyLast = automation.daily_research_last || {};
        const autoPromoGate = (dailyLast.auto_promotion_gate || {});
        const sprint7d = automationState.edge_sprint_7d_last || {};

        renderKV("statusKV", status, ["running", "cycles_completed", "last_cycle_at", "last_note", "last_error"]);
        renderKV("configKV", cfg, ["symbol", "symbols", "multi_symbol_enabled", "multi_symbol_active", "multi_symbol_shadow_symbols", "cycle_seconds", "llm_provider", "data_provider", "execution_provider", "auto_retrain_enabled", "auto_retrain_interval_hours", "auto_retrain_min_new_trades", "data_acceleration_mode", "acceleration_active", "entry_confluence_min", "ev_min_ticks", "max_spread_bps", "cost_model_enabled", "cost_slippage_bps_per_side", "cost_fee_per_share", "cost_min_fee_per_order", "updated_at"]);
        renderKV("accountKV", account, ["balance", "starting_balance", "daily_realized_pnl", "todays_trade_count", "drawdown", "risk_budget_remaining", "open_position"]);
        renderKV("goLiveKV", gate, ["live_mode", "autonomous_live_enabled", "passed", "blocked_autonomous_live", "reason", "metrics_lookback", "checks"]);
        renderKV("metricsKV", metrics, ["trade_count", "net_pnl", "win_rate", "avg_win", "avg_loss", "expectancy", "profit_factor", "max_drawdown", "sharpe_like", "hold_rate", "anti_decay"]);

        renderDecisions(decisions);
        renderTrades(trades);
        renderAudit(audits);
        renderNotifications(notifications);
        renderEquityChart(tradesWide);
        renderPnlHistogram(tradesWide);
        renderDecisionMix(decisionsWide);
        if (qualityRes.prediction_quality) renderPredictionQuality(qualityRes, controlsRes);
        if (symbolPerfRes.symbol_performance) renderSymbolGate(symbolPerfRes);
        if (ccRes.champion_challenger_daily) renderChampionChallenger(ccRes, ccShortRes);
        if (cbRes.cell_leaderboard_bootstrap) renderCellBootstrap(cbRes);
        if (inventoryRes.feature_inventory) renderFeatureInventory(inventoryRes);
        if (patternRes.pattern_leaderboard) renderPatternLeaderboard(patternRes);
        if (contextRes.context_leaderboard) renderContextLeaderboard(contextRes);

        document.getElementById("kBalance").textContent = n(account.balance);
        document.getElementById("kPnl").textContent = n(account.daily_realized_pnl);
        document.getElementById("kTradesToday").textContent = n(account.todays_trade_count, 0);
        document.getElementById("kOpenPos").textContent = account.open_position ? `${account.open_position.direction} x${account.open_position.size}` : "None";
        document.getElementById("kTradeCount").textContent = n(metrics.trade_count, 0);
        document.getElementById("kWinRate").textContent = pct(metrics.win_rate);
        document.getElementById("kPf").textContent = n(metrics.profit_factor, 3);
        document.getElementById("kCycles").textContent = n(status.cycles_completed, 0);
        document.getElementById("kNote").textContent = status.last_note || "-";
        document.getElementById("kAutonomyHealth").textContent = `${n(autonomyHealth.score, 0)} (${autonomyHealth.grade || "-"})`;
        document.getElementById("kDataReadiness").textContent = `${n(readiness.score, 0)} (${String(readiness.grade || "-")})`;
        const windowMins = Math.max(1, Number(qMetrics.window_minutes || 60));
        const sampleCountWin = Math.max(0, Number(qMetrics.samples || 0));
        const samplesPerHour = sampleCountWin * (60.0 / windowMins);
        const nowMs = Date.now();
        const oneHourAgo = nowMs - (60 * 60 * 1000);
        const decisionsLastHour = needWideHistory
          ? (decisionsWide || []).filter((d) => {
              const ts = Date.parse(String(d.timestamp || d.ts || ""));
              return Number.isFinite(ts) && ts >= oneHourAgo;
            }).length
          : null;
        document.getElementById("kDataPace").textContent = decisionsLastHour === null
          ? `${n(samplesPerHour, 1)} samp/h`
          : `${n(samplesPerHour, 1)} samp/h | ${n(decisionsLastHour, 0)} dec/h`;
        document.getElementById("kProjSamples").textContent = `${n(samplesPerHour * 24.0, 0)} / day`;
        document.getElementById("kTopCause").textContent = topCause.cause && topCause.cause !== "none"
          ? `${String(topCause.cause)} (${n(topCause.count,0)})`
          : "none";
        const llmRows = Object.entries(llmHealth || {});
        if (!llmRows.length) {
          document.getElementById("kLlmHealth").textContent = "warming up";
        } else {
          llmRows.sort((a,b)=>Number((b[1]||{}).fail_rate||0)-Number((a[1]||{}).fail_rate||0));
          const [p, m] = llmRows[0];
          const unhealthy = Boolean((m||{}).unhealthy);
          document.getElementById("kLlmHealth").textContent = `${p}: ${pct((m||{}).fail_rate||0)} ${unhealthy ? "(cooldown)" : ""}`;
        }
        document.getElementById("kWeeklyExp").textContent = weeklySel.id
          ? `${String(weeklySel.id)} | conf ${n(weeklySel.min_confidence,2)}`
          : "not run";
        document.getElementById("kAutoPromoGate").textContent = (autoPromoGate && typeof autoPromoGate.passed === "boolean")
          ? (autoPromoGate.passed ? "passed" : "blocked")
          : "n/a";
        document.getElementById("kSampleFlowGuard").textContent = sfGuard.active
          ? `stall (${String(sfGuard.reason || "unknown")})`
          : "ok";
        const sprintGate = sprint7d.promotion_gate || {};
        const sprintPruning = sprint7d.symbol_pruning || {};
        document.getElementById("kSprint7d").textContent = sprint7d.at
          ? `${sprintGate.passed ? "pass" : "hold"} | oos ${sprintGate.oos_positive ? "ok" : "no"} | q ${n(sprintPruning.active_quarantines,0)}`
          : "not run";
        document.getElementById("opsRefresh").textContent = fmtTs(new Date().toISOString());
        document.getElementById("opsSession").textContent = sessionStatusText(sessionCfg);
        document.getElementById("opsBlocker").innerHTML = summarizeBlocker(decisions);
        if (monitor.safe_mode_active) {
          document.getElementById("opsHealth").textContent = "Needs attention (model decay safe mode)";
        } else {
          if ((autonomyHealth.reasons || []).length) {
            document.getElementById("opsHealth").textContent = `${autonomyHealth.health || "unknown"} | ${String((autonomyHealth.reasons || [])[0] || "")}`;
          } else {
            document.getElementById("opsHealth").textContent = status.running && !status.last_error ? `Healthy (${accel.mode || "standard"})` : "Needs attention";
          }
        }

        setHeaderChips(status, gate);
        if (fastFirstLoad) {
          window.__apexHydrated = true;
          setTimeout(() => { refreshAll(); }, 0);
        } else {
          window.__apexHydrated = true;
        }
      } catch (e) {
        setFlash(String(e));
      }
    }

    refreshAll();
    setInterval(refreshAll, 15000);
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": MODEL_NAME}


@app.get("/api/status")
def api_status() -> dict:
    out = service.status().__dict__
    out["model_name"] = MODEL_NAME
    out["llm_provider_health"] = service.llm_provider_health_status()
    return out


@app.post("/api/start")
def api_start() -> dict:
    return service.start_with_guard()


@app.post("/api/stop")
def api_stop() -> dict:
    stopped = service.stop()
    return {"stopped": stopped, "status": service.status().__dict__}


@app.post("/api/run-once")
def api_run_once() -> dict:
    cycle = service.run_cycle_once()
    return {
        "timestamp": cycle.timestamp.isoformat(),
        "note": cycle.note,
        "decision": cycle.decision.__dict__,
    }


@app.post("/api/close-position")
def api_close_position() -> dict:
    try:
        return service.close_open_position_now()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/account")
def api_account() -> dict:
    return service.account()


@app.get("/api/decisions")
def api_decisions(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    return {"items": service.decisions(limit)}


@app.get("/api/data-samples")
def api_data_samples(limit: int = Query(default=200, ge=1, le=5000)) -> dict:
    return {"items": service.data_samples(limit)}


@app.get("/api/data-samples.csv")
def api_data_samples_csv(limit: int = Query(default=10000, ge=1, le=100000)) -> Response:
    return Response(
        content=service.data_samples_csv(limit),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="apex_data_samples.csv"'},
    )


@app.get("/api/research/prediction-quality")
def api_prediction_quality(
    lookback: int = Query(default=10000, ge=1, le=100000),
    horizons: str = Query(default="5,15,30"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    quality_mode: str = Query(default="all", pattern="^(all|good_only)$"),
) -> dict:
    parsed: list[int] = []
    for part in horizons.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if 1 <= value <= 390 and value not in parsed:
            parsed.append(value)
    if not parsed:
        parsed = [5, 15, 30]
    return {
        "prediction_quality": service.prediction_quality_report(
            lookback=lookback,
            horizons_minutes=tuple(parsed),
            min_confidence=min_confidence,
            quality_mode=quality_mode,
        )
    }


@app.get("/api/research/feature-ablation")
def api_feature_ablation(
    lookback: int = Query(default=10000, ge=1, le=100000),
    horizon_minutes: int = Query(default=15, ge=1, le=390),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    min_count: int = Query(default=20, ge=1, le=10000),
    quality_mode: str = Query(default="all", pattern="^(all|good_only)$"),
) -> dict:
    return {
        "feature_ablation": service.feature_ablation_report(
            lookback=lookback,
            horizon_minutes=horizon_minutes,
            min_confidence=min_confidence,
            min_count=min_count,
            quality_mode=quality_mode,
        )
    }


@app.get("/api/research/sample-coverage")
def api_sample_coverage(
    lookback: int = Query(default=10000, ge=1, le=100000),
) -> dict:
    return {"sample_coverage": service.sample_coverage_report(lookback=lookback)}


@app.get("/api/research/symbol-performance")
def api_symbol_performance(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizon_minutes: int = Query(default=15, ge=1, le=390),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
) -> dict:
    return {
        "symbol_performance": service.symbol_performance_report(
            lookback=lookback,
            horizon_minutes=horizon_minutes,
            quality_mode=quality_mode,
        )
    }


@app.get("/api/research/cell-leaderboard")
def api_cell_leaderboard(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizons: str = Query(default="5,15,30"),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    min_labels: int = Query(default=20, ge=1, le=10000),
) -> dict:
    parsed: list[int] = []
    for part in horizons.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if 1 <= value <= 390 and value not in parsed:
            parsed.append(value)
    if not parsed:
        parsed = [5, 15, 30]
    return {
        "cell_leaderboard": service.cell_leaderboard_report(
            lookback=lookback,
            horizons_minutes=tuple(parsed),
            quality_mode=quality_mode,
            min_labels=min_labels,
        )
    }


@app.get("/api/research/cell-leaderboard-bootstrap")
def api_cell_leaderboard_bootstrap(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizons: str = Query(default="5,15,30"),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    min_labels: int = Query(default=12, ge=1, le=10000),
    n_bootstrap: int = Query(default=300, ge=50, le=5000),
    robust_only: bool = Query(default=False),
) -> dict:
    parsed: list[int] = []
    for part in horizons.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if 1 <= value <= 390 and value not in parsed:
            parsed.append(value)
    if not parsed:
        parsed = [5, 15, 30]
    return {
        "cell_leaderboard_bootstrap": service.cell_leaderboard_bootstrap_report(
            lookback=lookback,
            horizons_minutes=tuple(parsed),
            quality_mode=quality_mode,
            min_labels=min_labels,
            n_bootstrap=n_bootstrap,
            robust_only=robust_only,
        )
    }


@app.get("/api/research/oos-daily")
def api_oos_daily(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizon_minutes: int = Query(default=15, ge=1, le=390),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    min_train_labels: int = Query(default=150, ge=20, le=100000),
    min_cell_labels: int = Query(default=20, ge=1, le=10000),
) -> dict:
    return {
        "oos_daily": service.oos_daily_walkforward_report(
            lookback=lookback,
            horizon_minutes=horizon_minutes,
            quality_mode=quality_mode,
            min_train_labels=min_train_labels,
            min_cell_labels=min_cell_labels,
        )
    }


@app.get("/api/research/champion-challenger-daily")
def api_champion_challenger_daily(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizon_minutes: int = Query(default=15, ge=1, le=390),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    min_train_labels: int = Query(default=150, ge=20, le=100000),
    min_cell_labels: int = Query(default=20, ge=1, le=10000),
    challenger_min_confidence: float = Query(default=0.60, ge=0.0, le=1.0),
    challenger_max_confidence: float = Query(default=1.0, ge=0.0, le=1.0),
    min_daily_selections: int = Query(default=10, ge=1, le=10000),
    direction: str = Query(default="ALL", pattern="^(ALL|LONG|SHORT)$"),
) -> dict:
    return {
        "champion_challenger_daily": service.champion_challenger_daily_report(
            lookback=lookback,
            horizon_minutes=horizon_minutes,
            quality_mode=quality_mode,
            min_train_labels=min_train_labels,
            min_cell_labels=min_cell_labels,
            challenger_min_confidence=challenger_min_confidence,
            challenger_max_confidence=challenger_max_confidence,
            min_daily_selections=min_daily_selections,
            direction=direction,
        )
    }


@app.get("/api/research/cost-stress")
def api_cost_stress(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizon_minutes: int = Query(default=15, ge=1, le=390),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    multipliers: str = Query(default="1.0,1.5,2.0"),
) -> dict:
    parsed: list[float] = []
    for part in str(multipliers or "").split(","):
        try:
            v = float(part.strip())
        except ValueError:
            continue
        if v < 0:
            continue
        if v not in parsed:
            parsed.append(v)
    if not parsed:
        parsed = [1.0, 1.5, 2.0]
    return {
        "cost_stress": service.cost_stress_report(
            lookback=lookback,
            horizon_minutes=horizon_minutes,
            quality_mode=quality_mode,
            multipliers=tuple(parsed),
        )
    }


@app.get("/api/research/promotion-candidates")
def api_promotion_candidates(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizons: str = Query(default="5,15,30"),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    n_bootstrap: int = Query(default=300, ge=50, le=5000),
) -> dict:
    parsed: list[int] = []
    for part in horizons.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if 1 <= value <= 390 and value not in parsed:
            parsed.append(value)
    if not parsed:
        parsed = [5, 15, 30]
    return {
        "promotion_candidates": service.promotion_candidates_report(
            lookback=lookback,
            horizons_minutes=tuple(parsed),
            quality_mode=quality_mode,
            n_bootstrap=n_bootstrap,
        )
    }


@app.get("/api/trades")
def api_trades(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    return {"items": service.trades(limit)}


@app.get("/api/notifications")
def api_notifications(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    return {"items": service.notifications(limit)}


@app.get("/api/config")
def api_config() -> dict:
    cfg = service.runtime_config() or {}
    cfg["model_name"] = MODEL_NAME
    return {"config": cfg}


@app.get("/api/symbols")
def api_symbols() -> dict:
    return {"symbols": service.symbol_collection_status()}


@app.get("/api/session-config")
def api_session_config() -> dict:
    accel = service.acceleration_status()
    return {
        "session": {
            "enable_session_filter": settings.enable_session_filter,
            "session_start_et": str(accel.get("session_start_et", settings.session_start_et)),
            "session_end_et": str(accel.get("session_end_et", settings.session_end_et)),
            "timezone": str(settings.timezone),
        }
    }


@app.get("/api/adaptive")
def api_adaptive() -> dict:
    return {"adaptive": service.adaptive_snapshot()}


@app.get("/api/acceleration")
def api_acceleration() -> dict:
    return {"acceleration": service.acceleration_status()}


@app.get("/api/economic-calendar")
def api_economic_calendar() -> dict:
    return {"economic_calendar": service.economic_calendar()}


@app.get("/api/finnhub-context")
def api_finnhub_context(symbol: str | None = Query(default=None)) -> dict:
    return {"finnhub_context": service.finnhub_context(symbol)}


@app.post("/api/acceleration/mode")
def api_acceleration_mode(mode: str = Query(..., pattern="^(standard|accelerated)$")) -> dict:
    try:
        return service.set_acceleration_mode(mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/metrics")
def api_metrics() -> dict:
    return {"metrics": service.metrics_snapshot()}


@app.get("/api/model-monitoring")
def api_model_monitoring() -> dict:
    return {"model_monitoring": service.model_monitoring_status()}


@app.get("/api/quality-controls")
def api_quality_controls(
    lookback: int = Query(default=2000, ge=100, le=100000),
) -> dict:
    return {"quality_controls": service.quality_controls_status(lookback=lookback)}


@app.get("/api/data-quality-counters")
def api_data_quality_counters(
    lookback: int = Query(default=2000, ge=100, le=100000),
    since_timestamp: str | None = Query(default=None),
) -> dict:
    return {"data_quality": service.data_quality_counters(lookback=lookback, since_timestamp=since_timestamp)}


@app.get("/api/data-readiness")
def api_data_readiness(
    lookback: int = Query(default=2000, ge=100, le=100000),
) -> dict:
    return {"data_readiness": service.data_readiness_status(lookback=lookback)}


@app.get("/api/portfolio/optimise")
def api_portfolio_optimise(
    lookback: int = Query(default=5000, ge=1, le=50000),
    min_trades: int = Query(default=5, ge=1, le=1000),
    max_weight: float = Query(default=0.35, ge=0.01, le=1.0),
    cash_floor: float = Query(default=0.25, ge=0.0, le=1.0),
    include_shadow: bool = Query(default=True),
) -> dict:
    return {
        "portfolio": service.portfolio_optimisation(
            lookback=lookback,
            min_trades=min_trades,
            max_weight=max_weight,
            cash_floor=cash_floor,
            include_shadow=include_shadow,
        )
    }


@app.get("/api/portfolio/optimize")
def api_portfolio_optimize(
    lookback: int = Query(default=5000, ge=1, le=50000),
    min_trades: int = Query(default=5, ge=1, le=1000),
    max_weight: float = Query(default=0.35, ge=0.01, le=1.0),
    cash_floor: float = Query(default=0.25, ge=0.0, le=1.0),
    include_shadow: bool = Query(default=True),
) -> dict:
    return api_portfolio_optimise(
        lookback=lookback,
        min_trades=min_trades,
        max_weight=max_weight,
        cash_floor=cash_floor,
        include_shadow=include_shadow,
    )


@app.post("/api/retrain")
def api_retrain(lookback: int = Query(default=2000, ge=1, le=50000)) -> dict:
    return service.retrain_adaptive_from_history(limit=lookback)


@app.post("/api/research/walk-forward")
def api_research_walk_forward(
    lookback: int = Query(default=10000, ge=1, le=50000),
    folds: int = Query(default=4, ge=1, le=12),
    min_train: int = Query(default=40, ge=10, le=10000),
    min_test: int = Query(default=20, ge=5, le=5000),
    bins: int = Query(default=10, ge=2, le=20),
) -> dict:
    return service.research_walk_forward_report(
        lookback=lookback,
        folds=folds,
        min_train=min_train,
        min_test=min_test,
        bins=bins,
    )


@app.post("/api/research/predictive")
def api_research_predictive(
    lookback: int = Query(default=10000, ge=1, le=50000),
    folds: int = Query(default=4, ge=1, le=12),
    min_train: int = Query(default=40, ge=10, le=10000),
    min_test: int = Query(default=20, ge=5, le=5000),
    n_estimators: int = Query(default=80, ge=5, le=500),
    learning_rate: float = Query(default=0.1, ge=0.01, le=1.0),
    max_bins: int = Query(default=16, ge=2, le=32),
) -> dict:
    return service.research_predictive_model_report(
        lookback=lookback,
        folds=folds,
        min_train=min_train,
        min_test=min_test,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_bins=max_bins,
    )


@app.get("/api/automation")
def api_automation_status() -> dict:
    return {"automation": service.automation_status()}


@app.get("/api/automation/guards")
def api_automation_guards() -> dict:
    return {"guards": service.automation_guard_status()}


@app.post("/api/automation/run-once")
def api_automation_run_once() -> dict:
    return {"automation": service.run_automation_once_manual()}


@app.post("/api/automation/run-suite")
def api_automation_run_suite(force_daily: bool = Query(default=True)) -> dict:
    return {"suite": service.run_research_automation_suite(force_daily=force_daily)}


@app.post("/api/automation/run-sprint-7d")
def api_automation_run_sprint_7d(force_daily: bool = Query(default=True)) -> dict:
    return {"sprint_7d": service.run_7day_edge_sprint(force_daily=force_daily)}


@app.get("/api/autonomy")
def api_autonomy_status() -> dict:
    return {"autonomy": service.autonomous_research_status()}


@app.get("/api/autonomy/health-score")
def api_autonomy_health_score() -> dict:
    return {"autonomy_health": service.autonomy_health_score()}


@app.get("/api/autonomy/error-causes")
def api_autonomy_error_causes(
    lookback: int = Query(default=5000, ge=100, le=100000),
    window_minutes: int = Query(default=1440, ge=15, le=10080),
) -> dict:
    return {"error_causes": service.autonomy_error_causes_report(lookback=lookback, window_minutes=window_minutes)}


@app.post("/api/autonomy/start")
def api_autonomy_start() -> dict:
    return {"autonomy": service.start_autonomous_research()}


@app.post("/api/autonomy/stop")
def api_autonomy_stop() -> dict:
    return {"autonomy": service.stop_autonomous_research()}


@app.post("/api/autonomy/run-once")
def api_autonomy_run_once() -> dict:
    return {"autonomy": service.autonomous_research_run_once()}


@app.get("/api/autonomy/self-scan")
def api_autonomy_self_scan_get(force: bool = Query(default=False)) -> dict:
    return {"self_scan": service.run_self_scan(force=force)}


@app.post("/api/autonomy/self-scan")
def api_autonomy_self_scan_post(force: bool = Query(default=True)) -> dict:
    return {"self_scan": service.run_self_scan(force=force)}


@app.post("/api/research/daily-run")
def api_research_daily_run(force: bool = Query(default=True)) -> dict:
    return {"daily_research": service.run_daily_research_automation(force=force)}


@app.post("/api/research/weekly-experiments")
def api_research_weekly_experiments(force: bool = Query(default=True)) -> dict:
    return {"weekly_experiments": service.run_weekly_experiments(force=force)}


@app.get("/api/promotion/status")
def api_promotion_status() -> dict:
    return {"promotion": service.promotion_status()}


@app.post("/api/promotion/evaluate")
def api_promotion_evaluate(
    report_path: str | None = Query(default=None),
    min_folds: int | None = Query(default=None, ge=1, le=24),
    min_model_selected_trades: int | None = Query(default=None, ge=1, le=100000),
    min_expectancy: float | None = Query(default=None, ge=-100000.0, le=100000.0),
    min_net_pnl_edge: float | None = Query(default=None, ge=-100000000.0, le=100000000.0),
    require_recommendation_promote: bool | None = Query(default=None),
) -> dict:
    return service.evaluate_promotion_policy(
        report_path=report_path,
        min_folds=min_folds,
        min_model_selected_trades=min_model_selected_trades,
        min_expectancy=min_expectancy,
        min_net_pnl_edge=min_net_pnl_edge,
        require_recommendation_promote=require_recommendation_promote,
    )


@app.post("/api/promotion/promote")
def api_promotion_promote(
    report_path: str | None = Query(default=None),
    note: str = Query(default=""),
    min_folds: int | None = Query(default=None, ge=1, le=24),
    min_model_selected_trades: int | None = Query(default=None, ge=1, le=100000),
    min_expectancy: float | None = Query(default=None, ge=-100000.0, le=100000.0),
    min_net_pnl_edge: float | None = Query(default=None, ge=-100000000.0, le=100000000.0),
    require_recommendation_promote: bool | None = Query(default=None),
) -> dict:
    return service.promote_predictive_candidate(
        report_path=report_path,
        min_folds=min_folds,
        min_model_selected_trades=min_model_selected_trades,
        min_expectancy=min_expectancy,
        min_net_pnl_edge=min_net_pnl_edge,
        require_recommendation_promote=require_recommendation_promote,
        note=note,
    )


@app.get("/api/go-live-gate")
def api_go_live_gate() -> dict:
    return {"go_live_gate": service.go_live_gate_snapshot()}


@app.get("/api/kill-switch")
def api_kill_switch() -> dict:
    return {"kill_switch": service.kill_switch_snapshot()}


@app.get("/api/audit")
def api_audit(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"items": service.audit_events(limit)}


@app.get("/api/research/feature-inventory")
def api_feature_inventory(
    lookback: int = Query(default=2000, ge=1, le=100000),
) -> dict:
    return {"feature_inventory": service.feature_inventory_report(lookback=lookback)}


@app.get("/api/research/pattern-leaderboard")
def api_pattern_leaderboard(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizons: str = Query(default="15"),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    min_labels: int = Query(default=20, ge=1, le=10000),
    stress_bps: float = Query(default=3.0, ge=0.0, le=1000.0),
    min_accuracy: float = Query(default=0.50, ge=0.0, le=1.0),
) -> dict:
    parsed: list[int] = []
    for part in horizons.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if 1 <= value <= 390 and value not in parsed:
            parsed.append(value)
    if not parsed:
        parsed = [15]
    return {
        "pattern_leaderboard": service.pattern_leaderboard_report(
            lookback=lookback,
            horizons_minutes=tuple(parsed),
            quality_mode=quality_mode,
            min_labels=min_labels,
            stress_bps=stress_bps,
            min_accuracy=min_accuracy,
        )
    }


@app.get("/api/research/context-leaderboard")
def api_context_leaderboard(
    lookback: int = Query(default=100000, ge=1, le=100000),
    horizons: str = Query(default="15"),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
    min_labels: int = Query(default=20, ge=1, le=10000),
    stress_bps: float = Query(default=3.0, ge=0.0, le=1000.0),
    min_accuracy: float = Query(default=0.50, ge=0.0, le=1.0),
) -> dict:
    parsed: list[int] = []
    for part in horizons.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if 1 <= value <= 390 and value not in parsed:
            parsed.append(value)
    if not parsed:
        parsed = [15]
    return {
        "context_leaderboard": service.context_leaderboard_report(
            lookback=lookback,
            horizons_minutes=tuple(parsed),
            quality_mode=quality_mode,
            min_labels=min_labels,
            stress_bps=stress_bps,
            min_accuracy=min_accuracy,
        )
    }
