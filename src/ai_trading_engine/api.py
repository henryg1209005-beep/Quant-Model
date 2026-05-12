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
    .blocker {
      display: block;
      margin-top: 2px;
      line-height: 1.3;
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
      color: var(--muted);
      text-transform: uppercase;
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

    tbody tr:nth-child(even) td { background: #fafbfd; }
    tr:hover td { background: #eef3fb !important; }

    .kpis .card { transition: box-shadow .15s ease, transform .15s ease; }
    .kpis .card:hover {
      box-shadow: 0 4px 14px rgba(15,23,42,0.09), 0 0 0 1px rgba(15,23,42,0.06);
      transform: translateY(-1px);
    }

    .flash:not(:empty) {
      padding: 5px 10px;
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: var(--radius);
      color: #92400e;
      font-style: normal;
      font-weight: 500;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

    @keyframes pulse-dot {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.2; }
    }
    .chip.ok::before {
      content: '';
      display: inline-block;
      width: 5px;
      height: 5px;
      background: #6ee7b7;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: middle;
      animation: pulse-dot 2.5s ease-in-out infinite;
    }
    .chip.warn::before {
      content: '';
      display: inline-block;
      width: 5px;
      height: 5px;
      background: #fbbf24;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: middle;
    }
    .chip.danger::before {
      content: '';
      display: inline-block;
      width: 5px;
      height: 5px;
      background: #fca5a5;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: middle;
      animation: pulse-dot 1.4s ease-in-out infinite;
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div style="display:flex;align-items:center;gap:14px">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKAAAACgCAYAAACLz2ctAABG2klEQVR42rW9ebxmVXUtOuba+2vOOdXQWghVFEgnCkhn9AoRJEEDijGiN4L3pTGK5uYl8Zqr9xpDBJXEGNP4YmKivwQw0agYjIBNFBQVE4kINs/fUwFFpVGqSqo7zdfsPd4f+2vWXmvOtfep5Nbvp5yqc87+drP2XHOOOeYYkuVdggIIan8EAIHJ/wff/D/yp+3nrOd86F3Nf+4fCT4B/6F7lfi9dR2yzQ8zesLzryU6hATXKNE1t30G/inK7FgOnN9KUS/F/0j/v7Q/JPpnBg+OAAl9ybP++9HHiPKR9P7DxnOof187Fzbe5vpPBAud6/j8aDkHPyfauVjPQ1p8lnLPyPl/J59nHYnRWTN53wTinVcc6Jx2G/3YF0eQ+A2S2o0nZPoz4v2+v0YggIh31Mm/UuLXT7yFIt7ni/EAGdw6QSIKcvI2au+51Bdp7cVQvqb3dXierIcVSS1EavfZijsSfS31J2L/HsWLRjL7mOnvt3pv1EgqRrSsB7LpJzgmLkb/sGBF+x8iwcnXHnT8HkVxUbzfJuvPUlIvw/RhSxAERLkWxhFL9Guef2T4YoRfSxU5JNw/vPPy7gFTi0MNPWF0E/OXGC1fMZ6o90LL/F7RP28yGUEZxLlW26/3OQThJHwOVLZZ0t721CAtRrSMt0oJIgmh7GbTkwxuCCnxopxGKxF70dH7ucRNprYY2RBZUhGKxlEZ3Dv6589W2WfTNTC8VgmDiRLlRdTAoV0r1TMU9dTFu1YXrWkRb8sQ5e7TWxUMAjZbPEb/jfPeIvEWk7Xgp28s40BUX/vKQ6J3XJH61tOQx7QqNvwtWI26id1Fwogs7dMB65yoPBOZxirt5QsXizQGtJZlR/T8/djp1G0jfN0Z3BiZJq0SrHSxcy1YoU2JQuri8N5YMZJwsY45zXHEe4B6jiXawoqeixI1potaK2i4nooUdrHk3xuGxVwQTSXYHVjlffSjsYiez9HY4dYXfBtzx6oKroX+8IFIkFRznsCayT2DfdT7n3+z/ORcwrxDfzPFTIAlWDpaNcrg5dKiZJQYeItdu/fKCyfhZyYWZjJSIlr5UnvZxLt3/i7BevRT6wMaab7/PMTe/qU5/kkCJvKvxKmJPZl+UyXcqlhfONa2Nr1Zap4WVGXmNuhXuuF37MXrJ9p1CMa/8eHNs3E1aYRUUH+xRBrWmgS5MvUgSD+H5uRRhdE+vG4G/5M09JPKRsLoyHDLl4ZCi7V/cf7FzB6SpMBcrRr0LpphRGUDlGA9VdFxvWiBe8emBAs52KIYpBTewpTgFklia1wfGGs86LDIYLg4JF59Eh9SRKncos+2diLM4Zio8qWXK0N5kesvrZozquhKvSp3aIRhJNquRNvC6C/eIJdkPcektMkVLXhE9CqMfrTxIrEoL4yElSC9K2OQtinVd6LWRBKcleQWm9zWRInk4uXmPLCyYLbQqAPFjS8WjWKFbFWouPlzY/LNJQm/hqoDv1RCtMSLcvJ3Rxpb2YG3zETdVSQBF/mpg8QFC41oxTYdDVFeVkmnNURiu5e4XRZ2coSJzpAg3ulQT4MkgIHMZ8E4/VKzCWnVWXKiou0KnCl+pFC2ZT/q+MkqqeRqooDRaJlrMb4mKsCy1DdMsUo4C8sR42zUaCOtFmNyIxe2AnyjAsKHlMJ7LYzBdmrnFub40qIdKWq+T7VjJCbi4eo3W4L8K8zDpEWkskp7tghwjGCAenrNxCLzQ38MhFOtjSRIpqnkU237szpQzcb7FeJ3SGN5qSaAiNGxot6hUaNz/Izj05DJjsgorxM/QovyWVEv2LrIqCCgjk1JA0gkjI8j9UouuhlCNSLru5/fUgqzcgWtFwsnpNLpoQEEiwHjMNFK00pa8SK00nJRc2U/b01UzbRwxKCkZorsELTxvHMTiXdNqi9Gutp29bdGAx/FTpJFmk9a7XSEWwKNCGCBq2E6wpgpQ6sgQK35rl+vsvBUEoT1wJkoKhi02owlSgl2hPCaxOtAKatemEgdJMj9ghcqvGU+tkgr2IiBE/rIQ/xCOX21+iFWwfzUN7oeomhVfoJ6tAmrB1EATzFoTuJFNT9vEYu8IEr7PMhTrChrVXt+XiZitOmUIifaFrWoLiqZQV2ohJEuJbIjBr1xJV+fPSsRHbtN5r9anln/2plIdhRiE+02aom7/waLjsyniKNkerHLdCtAAMYmKk7SINt6LTphgBkyWDSidBPEjoRm1yHsfXsgPSXYGqn/bFj8SUOvLMT8xIuWEkRFagWZNKQVqfxUoyqI34pLVHSp/ndUsrMeNRhyqsIFSbu3Gr7V4j1QztuC1cspCmWJMXQRFWLUW1X+lkyt66PkktS6RNJQsDDuXAjtpogYWx9bNmkliOYhXFX7niSItRIvOlGKGMLMYwlAsrzH9kz2AFMSb0FJEyuLCfqK9l+sg6KfIM2SSjU2JR8yyEcZRBYl55QgWoWdgVZMedq3wH9JhS2HAcLrFINDmDheiHsygNGk6dqsIMbk350NJFoVIerkS/UEtGaKtOfPTZgbtQjDFGQjCvUqRFqkzgYWRpXdPE9m/Y2nUdWJKNsO02RcakMXqXxNFCwt/Egq+KTY95UK9itBESYM2nFsaOAYLcTkYkSYA7KBbKr0dsUCPBJzFtGi0rZgJcmnxrRRLsyDY6Q266BVflJn02gYlvjRLzj/YOyAVF5apqpsjyoWIhE0ci4J+vbiBQaI3bWQkAQStlHFZhNpkI6K1tKYA9J/3+lgpU530jj9qLFfreGYoLSPeHkJBovYqLue4Pv4mAK3RIzvIF8TozquLWAxuyUiQYRR2c3SAMK34LVSElE6hD60Boz2ElGhZSFBQqGRNjGYSglycQ/VcObVNvQ82YjoW7CGH9qh5ymiMViooOsSs539SEZjUZNKBES81VISTJME/UqaIAlj4RE2wuBje4Te9tRyUWpczxAPpNKiU4oPphJbSayPOkbobxpOJwQoVHWGI3YN+Zw05CNmqBd9wUS9TMZs55DNK7QrsaaCkVru1oLR7EMoNZIAY/A8jEKi5HhqP9rbDWawjdSr81pnKIhuotDTJLVtw5jzOTACic/bdDBpATFAzHChUCvHpYFGFD4wBh2MVI8zZPwqAz0h1YrBTeTkrZ8ehwG0EuZDEpBsSTsC+AuOokcl8RYbA1oVQ0ZyiOMqAPd0JxGjz2tR9H0UQzT0QGnDyn/WkP/8+pxd+rKhxRLkcoKY0Wt2FQLmhfjQB2vAMKOt1HhA0ZZFPb+RMMcJv9baLQzmaHXsc5b/Re1N0bGpCPCVel874uxReVF1wFqsvJnQh/ijQTQNcGeCltXMAdR2D5f8JbK1zEK86MRA5gPE32+Is91w1nybYuv0rMZ2oUGVt7oJYuCCGv0uiiqis28APaow7P1qcx9anVfPqRk9K+oDRghJrbQRj7ZpF9PVgkFGMHKbsGJScxSxCQNaY1+8IXJ6WJjKWUMUFWc4Xhi1aHRQwk6ABFuvv8XRaulJnA+RRjA0CKxClSAq6uJSesaza5svGlFTHDYwdrU+rYYopPq+iY6HaOhIeCR/C6ZOQq2PYFLJixjPk0Stnbivq053aYskTJAlxSoOKfkG+ybcxrXo4uenVFoWDJUYpAVHD8oWJwpiJkaXQlQygzos4AP56+lL09L7YfNuKqnflcRQUpS/GVuuGKufUmdlaEPWAeOFGvHAX+hRs12M2VWrp6nNphijmvQXfICtWcoAVlcnUH5SBZfazNcxWEwh06em8KD1li0UwoCVonkQA79NdkHF2IvDFK3+0rkkQ1adkgu3Bb9vKa3fGsKYS5CmXJRxa28KmZjnIDFZoEYe9V4kaUpHQmySCuO7ImeSByLXphR0DGc8GHR8xBiMSSLZugxLRLmnCZyb+bl6pFBUQG3FidJjbQBFTQU5Ud4ghQFs0Lzqui8GQYHWMLjOZhafVyhSj34SzDAwgVvWaEuafInUPy8aUkoN+8g6JubajkZoeJ9WpIuSctiMZprbLQMpDnsWxiX5dgw7FFJXR0BqkdCOnlRONqA8iShRllSoVqmhICoDMxNMs8YFFEP1yqD+a9cJi4liq+nZUcPbXiOoKC2qlRQ0kpTSGVrcT4lnc8zChK2k/VySUClU5CqDibfabIG04MFJmhnjbYekou2C4IFQYiCahlZgNOrYQKI0hTQVGIc6pUxaQ2IhMVYD7LWdiooSgRjMaWNmh150FCs4JFSvGiRV/B1VIhgGiqCjiDmnyhAWmAGo4ZbLRIKttXI8SEM4SalojKCKMvMhXlEhQXdFFM1Axjko6+CBaFWyOjcRcCO9W82Wui915o3B2vGLpmB7FDSxlsP+eohzavMe9Z2hUWMJCUEqI3g7NSGg3/IymA80RIlCIR6yPl8ijLsl4cKi3jlSCZrhtunz4oTxWGi4jYvGO5TJadHOBSWuisW80UyPWTJUGmjaYyXS2yGQYBbRECai0ppLVYP1a5AQ12UCzqGeFjh9MNmQvaUy3ymMmRWa5BkNpkVtpiFszQU0dTVqJAigIemFCTkJUVShqDT9VaqTrx6RqgBjPVEB7T5/OBcTVf9hkyBROFLhIUb6PRZFTofjCKPNWRuUT0xVVkNJSunOgMcmqWEfMdSxAnxJwaUYEim1fIzhTWyjZUodTK7l3dTznZSEmYRbGBUGUdPupPAoox2iDTE0zIE16hQTZFUlCjK8SUwA1YnKnBrxlaqKtDMZt9rJCW15idpgEIMeqh9F5lughHJjYlReXo7E4BjJik1gU6ooiVkLRUlKtNFRsW++Nvuh7kas60wr878pAkQNh5UU1d9iV6OeO7MNxY4thXritianA+w0YRiNRUIF7VYAWmF6Ii1i5GoLwmqNeR0GkYaKW2m8U9vapHFmQdddCeU5mLj5qbSKeoGlWkkkuiWi3PPGBUO9zy1tSJPS0ptEDIWweQ/bRVgWE7MBNAie4rNbJNFRCapTn5SggphajxhpPl6NTeJz7gyKummTAL1SVvEuIy9TCzip45AicS9dFFUKkQRn0ygeaGktysQSI/H+wipA1wOGs1Hb183TYqoRJ9Khs0TLpa7GKbBmLxgotCt5ozWMk7SSCIarGVonBC24qFrX9HMk5heqnZ+2viSWGJa/1VOBfJioatFiGBxRe48C3SVJ11UJpEDa4jFi4qfTgs35gzzSCmCkrcDpkRqpgszW2KEkpMGULY1MU8d8SlPUMBFjIJyB1C/i6TyJRYX0/E8ZvI/6uBYQj7jg0jowbBptsHrpiQFua4xisrMwyYdkS05p/bOcvn7FqERDRD3UDwk14djQ+kECRjH0VQh7IksrDiRgF4vCjWNqzkHq+Z4YzJyICWNMktGIJBL0V4VIqsJKSvbXULdPguGW7nc9BRBTrk5a9LGp4YCJKSi/BRZyAVWBSsOpiMpNVTEsy5pKsUNQHJviKKnAN1b0bONBQIM1IvVIyXBMM/hZqsVPonXXyHRnw89J2l6L1CO7aNOQbUBqpFk4kT4gmybyLa1AGg9XiTomSC8JJSsaM6pMqBYE8mdMSJDVhtabSAOi0J2MKEJjTZimiAluX4ThMTk4Fne72eDxRmUUQRIvJW3CqVh3UczUxYXFQ3JrjBrwFnVHDOO9MIpIg5ijIT1bW/ySEAOXGAeUIGqJLvshyVYSFbMY1pXjWxWMSsSJmNpS1woMWdmNVmOSGNpooukriIdP2Qq1vin6Z1O3VASAPLa3QkvFdAl6+rTfSFVoxwNQQzRfUrYMCAbYJRqOokwkOaQlbCA6KC0icOIgTgLegUyScqAsSxBlIlqIIkipTMpJwjlJGO8UCUEgSbJjDEsvMWATsSp9BkNQifzBbN8J8lZChrUbCZ33JsEUvsReHHVxKdb7yQzoP9qYQMiPI1VFBpHghoZ4mNj5kIggcw5lSYxGAwAlmmZcO90eRByKotAtsMJxTc4ranFulieS8955yVJXu4cyshBBkVoHS9EjFCRw1fCZt7GTaOccUI+A0jTtLzptSbTh63CGtT5nIOqcLNLilsrnkxbcRoXeJQENX6esO3FwLsNgsIoxSgAO27ZtxclPehJOfuIT8bgtR2BxcRHdbge7H9uNXbt24r77v4t77rkbP/zB96tjZF3keYZiXCqdvbhaJYBiNDQWuUPe6YFlUe9Byzrl60RJPyJqvMSphYhuqKMqSEiDADuVMd7J4bOsy/oFWqvVoHVrnQArnFMaWj1NrGLlZoa6eKoWIU0mjkCQZTkGgxUAxNatR+PFL34Rnvvc5+KUU0/Dlscdlny+j+7Yhbvvvgvvf98/4gMf+ABGowG6vcUqGiYa/1mWYThcxTOecQ6uuebNWFlZg3MOILG42MenP30brrnmGuSdHsqy1AWemkfVGkTDEz3maI4mpYNopRpBoSpx6iFZ3qWeIzBum1mLIOghc7oNpvKPVF6UiFRqhPb18ZSLVAkFxGz7Gw1XcdRRW/Ha174Wl11+GR53+OEAgLW1IdYGA5BEv9dHr9edHWY4GmM4GMI5wdLSAgDgrru+giuv/H188pMfR6fTR6kB05OXxDnBeDzCxz/xCfzccy6Mls+Fz74It376X9Dp9ucLMEoj1jP01GQlKy09nxvA7EaT7cA9Pst7jPVTwu2WejFBCbTzpMXFimHvZS1CpofeNYVTa/F6x3SuytnKYoRXvOIKXH311Xj844/AYDjGcDhAUZTo9/ro9zsoSuBHP3oEDz/0MHb95DFsWFrAoYcdjq1bj8LGDUsYjkvs37cPhxy8GSDwu7/3BvzhH/wBOt0FcErI9Vp6WZ5jOFjBxRdfgptv+Sj27t0PJw4lC2zetBF//w/vxy//0n9Dp7vgLT5tm9Tk1aRhe7R2AwsCaulxkpSho6KiIdMF2GWkuC4pNaTgYKZMbdu3iLE6k7HNS4T8aFK26beOJFzmUI7H6PW6+It3vhO/9rJfxXBUYG11tYqKADZsWMTOnbvw/ve/HzfddDO+/vWvY+fOXeCkOFhcWsKxxz4Bzzr/PLz0v/1fePrTnoqV1QGGgyEOOmgj/uzP34HX/I9Xo9PtgyU9j6iq0CmKIW77zO0475nnYnllDU5kggcXOPfcZ+LrX/8q8rw7KUaMHYpJ1+7EjtaUtzX5UYgRkbGu8VOBAC7vMcu7zPIes6xX/Xf697zr/Vuv/m+Z/3U3+N1ucBzluNG/hd8Pjln7uht8Ts84XvC7WZedTp/O5dywtIGf+tSnSZK79+zj/uVV7t+/wrXBkCXJd73rr3nMMcd4AF/GPO+x0+0z7/Tpsu7se91uj6961a/z0R0/4XA45p49+0iSv/F//2b1/d7i5Jx77PWXCICXXvpiliW5d+8yV1bXuHffMknyr9711/PfmZxzfF/9v1f/lqs/01vHfWr+jOg5RL/f5vjzr/O8x4lIOe0WjSKgXXUztTJdUiVqYz5gG7tIPdqxhVSYkjxPo49zghtv/CdcfPFF2LN3Pzp5jqIs0e12MRoN8KpX/Xe87x/eCwDo9RZBcpLPAWQ5G1aSSTExLgqMR2s4/fQzcMstt2DLEUdgeXkF3U6Oc859Ju65+y50e5PtlCVc5vClf/sSTjvtVCyvrCLLMjjnMBys4ak/9TTcf9+9cFnHo/jTuHUBhGQysOPIJyr+35R7Nz3Ptls1Qka0wuGnBuEwodag8NrWTalqQboxUX3qhAYvn3FZhtFoDW9729tw8cUXYfeefeh0OiCATp5jdWUZlzzv+XjfP7wX/f4SOt0FjIsCRVnMczlvKy9JjMZjkCX6C0v46lfvwXOfewl27foJnHPo9/v48z/7U+R5B2VZIHMO4/EQl73kJTjjjKdgeWUNeZ6jKAos9Lt493veg3u/8y10On0vbzTcphSkgWCC41hfXERCbYJKrt5IkUv1syVwfg+LED9HAiuyolgeENRnTVOLg2iwM2hXMdlJtcRFi9Tf+izLMBys4AUvuBQf+ciHsW/fMrI8A8vqR3u9Dl7wghfilltuQn9hA0ajYYvPrp9Dt9PB6up+vOjFv4gbPvQB7N23jE0bl/CCF7wQH/3oR9DpLqDX6+GuL9+J444/HmtrAzgRuCzD7t2P4eyzzsIjj/wILstnk4S1uS2Z7DwUXVRShc4kjd9p0sdq8DsQYUqay5F1x/TAwkoCXp05qwGl7ymB86S0MFRusjQ16OLR1BUV4czqfSuKMQ499HD88dvfhtFoPHsTi7LA4mIfb37LNdXi6y9NFp/eTpKE+tNwNES3t4AP3/BBfOxjn6gq5OEYr3nNa6rjDlfxipf/Gk466USsLK9CRDAuCvR7HfzVX/0VHnroQXQ63Umh4xEKJGBRq04D9Z1H0KTaNZ3IU9jWIoqSmD5qKS1QcFHFPOhHQDRjbrQmqzRFUjmA6qopzwhN/MRwYGKQjgqyvIp+V175RrzpTVdVeV+ng6IosLjQx91334NzzjkHAFBCgDLRcgxNnQVBpHUYDtdw5pln4/Of/zxIYmmxj/PO/xnc+aV/xbe+/R0cdeSRGI5GcCLodHI8/MgjOOvMM7F7926Im0c/Lc+L7sV/hmJu9FnhbLXVCoU+cWfmmnHjwTHpCSJxMcIEO1rzujDVUdczYR/Ml0ym7RhN5tXZ0DIZfhmPhjjs8C34tZe/HMPhCHmWzdtvmcOb3vxmjEYDiMsAlol2ZJ3RK8pwelGU6HT6uPsrX8ZnP/MZLC0tQETwvOddjCuueCWOPeZoDAbDKh8sCnQ6Od7+9j/Brl07kedd0AedBYnpP9ENgloD0lTobgywX8bioyobzJa1pUlKljAHtCyfAqBZrZY0hD7V+rGA4xaoOlknwIYevcG551mGwWAFr3jFK/Hud/819u5brvq1RYmlxQV88Yv/ivOfdT6cyzzTQ0kaVFAb/pb5vZt+5ote/Iv44Af/EaurA+zbuxedTo4NGzZiPC5QskS/18O9934HP/W0p2GwNgBEwJIt7kdTx6JFdaoZTE4YPiIW6G11nNafD07voyMNlF2ba0XCqyPij4lNdydiG6iw4otk1lLX5OUqQQVXliVEBL/wwl8AWRktz+RhneC9f/9elMUYzuUNXEjD19DPOSfXUpQFxOX45Cc+gfvvfwC9XhcHHXwQlpaqxTd9vp1Ohre+9Y+wvH8fsmnhETbrIcmZHAl9WYJ7JhYj2VBfEAnEJMlg3FabOQ5zQtqyc6wzLp0axcO2mlCBQjQRG6UgsXwqCEVnWXEjUsWTaGhX19kiIoLRaICt27bjaU97OobDAs45lBPM79EdO/Hxj38cIg5lWRiukPUbX5+K0FgeVRTrdLrYv38vbv/sZ5FnDmtrAxTFGOIEZVlF3y/d+e/4wAf+EXnew3hK5Qom2NlWiG1y/xgYBKmFS6g4FimdBhZiVJi7GntJxBdjMX1fpC5SbolNEi0mqwNrpgQ2lxq8FrSnF/mVtUgyiXau+vtZZ52Fgw/ejOFwOOHaEb1ujjvvvBMPP/Qg8rwXGHRLUiXP1t0LxIpEcMMNH8RoVKCT55AJ24WTc3vrW/8Io9EQLsti37mk6zpM1TKRRNEnxs4mDZoi0qYv3HyumkizU21gaU2v0axsadKqjMFuCeYE2EZPWZnnVbmE04+ovj7zzDPmW4M3ufflL3+5+lHnGQ16i5gtpB/RMDX02O49GI3H1eIDUJQlNmxYxO23fx433/xR5J3enLrFhMt6eu7SCwCasQ5tU/Cg0KHvsgTDD9pQ+ZIgl7Tv0nyjdhIxZRHv+Wgagm7hIaHawYvSbZGEKx0Dwit1/HG6gCafecLxJ0SLmADu/srdtbdcBAdYUcasX7IESfyv170Oiws9jEcjcJKPFkWBt7zlLSiLAs5lwQsTMsxTkhhMazKHpjmhCLyigSdhQJCEwCgt6Udp/ZK62HdDFFFuRbqChswZqd8MEd3zQxt2CpEjNkjC1lQQ5tdQltVxjnj8ltrxqyp1iB/+8MHq18tyPcqLDcNAVddlNFzD05/+DFzy/EuwsrKGTqeDsiyxYWkBt3zs47jttk+j0+mjGI9nL07cHjMiHw0N71rkZuzYJNKg3iqKRxzaCWsmtQAta7XpXLBlciKKOoGqJqAoF3DaN1bmGiLAOmFfGjlcplhXgSY0CZd1sLS0sfZveZ7jscd2Y8fOR+fNoJQJSzIHVNr/JLKsg7dccw26nQ7G46qXnOUZ1tYGeOtb/wiQ+SyI/bAD6zJC0YpJuaIzma+yKZejgt2SCf0by2lAjHyfgUq+JvhExbwFmsGgJttrqGK1xpDE8Ao2gG5FmSrLstl75jy4ZGVlBaurq4A4j3UierTRsHzRAfcscxgNV3Hpi16Mn7ngfOxfXkGWOYzHYywu9PGhG27AnV/6IrpZhqIs9ZllGOCuNmhVE4UyDL1peCM33fuwvVkTk5IGQ/MWOwdrTkmKAqfGuggFD0Xp0Vr6gi3VlaKEOjDYE1U7UN+uyqKASDnngJAVAcChGh7SgFdTcxlJ+1qR6vP6/T6uvPINKIqyWvQiyPIc+/Yt421/9Laq3bZ0BIRFPN0XSdd5wlGRP4vXJ/Z95YgEZNJQw9KST9Ze/Ha7hp7RVIHARcryauNfWjjhaOqamkZLyvwubFlrL4N495cKYOJPugmKYoTl/cuznFCc87AtBpWb1HD6RiHQMKF2GcbjIV75qt/AKU9+ElYmDGuWJZYW+/jba6/DN7/5DfSPfQ7cs99f5Z4hlV3RafbPSzRL2dlLVGcu6VicqCA1wx0s9GE2PV1ouAkYyhEitU93s3xN81dTNeKoJ+Ba9yRMSgl93LKJIaMphEqwUJQQP4U+fjApNtysX1ug0+1hw8aNte1XRfHZwD2cRB4Rh/FoiG1HH4vfe8PrsTYYVbGLJbrdDh57bDf+9E/eDhEgf8pvYeGYc4FtzwaLISBZUNSF8ifhALjoRoc0ICPG8PCs6KEkVGPRkp4fkyKkIcjUNaJ9c2ekTIwlmAXWIqHovVvfv41Ns61KnqPOCiuESImFgX7w/e9VEZAV1DIajnDYoQdjy5YjABQJTZwGxQJfZkwEZTnGm66+CocddihGwyHEOYzHBbrdDv7sHX+BH/7gAXSP/3mMj7wA4+UC3af+HrK8B7DwOjzBS0ct/zWijVj/bjkWKYQGMok3JkWOvMVOU+sHgVmh0IhemuYzdMM/CwyGBDrRTX5shiqppCRfQ808D2+f3IyvfvVrc0IngaIs0O/3cPzxT6gKB+cSGndsjAaZcxgOV/HM856Fl770cqysrCHv5ACJhX4f3/r2fXjHn/8psryDhbPegJw5XLGKTceeg86xF6EcDwDJgkKf6RejlhaLIhdHw20zkU+IYetKxReF1s4jjapY4gUtF9khIKxWmbBChd7rbWzHpdpz0i7kq7MGdWYOyxKQDP/2pX/Hjh270Ol0ap929llnzSOlMCHjZU/3yWSccnFxCX/5zndCnKuq20mu2u3meOPvX4m9e3bDFWMMfnw34IAMBVgS2RmvmWzB5URzxvDe06xjqZEDptIklodzKnrSNk+UmCCrCsxLs98wkz4htIgAijiipZfnY0XU5ot9MmXKyCR44LS24dCStYqeJYlut4eHHnwAt956G3q9CgyWifbLBT97IXq9RYzGo4SyV5r2nrkq9/v9378Kp5zyJKyurMI5h9FohA1LC/j4J/4FN9zwIeTdBZQA1r7yVpSD3XD5AkaDZXS3/TT6x/88OImC9R1JdIkNrQeteTjKevK5aYHDFl1cxPrfQXoiLbtIrq57TLsFbFpsKeGzJmhpqD1JrESlRk7GVTASTpsM7vz0q+uvv65qfovAOYfllVWcdebpOPenz0UxHk5IqkwocsVdnDyvJD0uvPA5+J+vfQ2Wl1eR5xWtKs8y/OSx3Xj1q397frh8AbL3Acj3boRb6CJ3BTqO2PCM30XW6QPlOO5BM5BESSp+MdV0aFBAZO3eiYly+OQPKrCR5l1HRfpuGgFF9G3S5/uL6LYEIml7qGBhU/v36N0K3Yhou7mbFzf/My4K5HkPt952G+6444tYWlrAeDyuuIEQ/NZv/aZ67ckBb0GlJ7O2jBNOOBF/d+3fYTQqPMm+EouLffzO7/xP3Pudb8/lNVhCxGHtq3+O8XAZWdZDOVzBwraz0D3uYhSTilgk0LumtJRKNSj6wgbdQBikC7FbjhIGhRCCs6wwpM6GiQik3kASTd04sUf/Zs34OFfR0wzadlNRUmx5EKcKCMJlGYrxEK9//e9iOBzBOYc8y7C8vIrnX/I8/OrLXo7BYAXdXs8Lsopi6GT763S6GA5W8PjHPx4333xzNeMxGE7mQUbYuGEJ1157Pa679m/R7fYnbJcJbJL3UOz4Bnj/jcBCH5krgLLEwk+9FpJ1q4o4hEXEX4iakCjV4Z8YEmnqiMTHFWly3tTgK2l4NvPfyFyWX1U/VW0wXRLDRkouwMBpUqhw+BSAVFoOQUuTHUD9ZpBEp9PD9753H7rdPi782QuwulqpURVliWdf+LO444478N3770Xe6U14hNP17yanXZEMnAgGgxUcf/wJ+OhNN+GUJz8J+5dXkGc5RqMhNm/eiDvu+Fe89KWXgZQaiDwhIEFYotxzP7pP/iV0sy5QDNA/7AnAT+7F4JG7IVnPk2zzqGuiKJoHjJY0k1EUJ07LLkLsITOB8ZnU24BijwlkzmVX2eySAGRky0UiDbL/qskMDO6ghjOKUjRZxjFzMcgs7+L2z96Gk590Cs484ylYXV2DiKDT7eK/vvi/YsfOnbjry3eiKMYVQ3rSJ2ZJFMVo8r8xLnvJZfjQhz6EJxz7BOzdt4xOp4PxeITNmzbiK3ffg+dfcgn27NmHLO9MRiz98yqBrIvR3geRbz4WS9ufCinW4PIO+oefhL1fvRYsxxBx9ZxPJN3YbzunYUEtUNqxEne14rOgbZQzjaDBuUl9AeZXzbZI/4MpE9dybRpfmqfGaqOTKUdzj2UjRtcj1WesqY96WwIRESKmX/7zP/8zjjvueJx99pkYFwUGa0MsLi7ihb/wAjz96ecAAqwsr2JtdQVlWU2ubdu2HRdd9HN42x+/Hf/7f78OvV4fa4PBpN1XYNPGDfjCF+7AJZc8Hzt37kC3O9X1U0ZWRSAswb33Y+PpvwyRLqQcwh10FIY77sfwR1+ZREEmcEhN5lcOkKvc9NPSiuds55XhoJu3GF3WpUAUWr4hXhhNniVoOQxcuX1TZ7GmzBSsLWq5SbMptmgyI4SIA8sCZImrr74ar33d69DrdrC8vIrhcITNB22CE2D3nr146KGHsLKyiqXFBRx51JE4aPNmAMCevfsBEnmng6XFPgDgHf/PX+D1r389VlfX0Ol0J4sP5rmJZOBoFRuf/Q4c/IzfguzfDXQXMfzJffjR3z0NKIaVmyipa3j7utFsosw3KxTU/639xJskWeKSUDir/p65LLsqjlTGhasAtNQxPRrOkpquSajtF+GPmrazJBRVA+O9iMo11QbM4FyGW2/9FG697TYcve1onHjSiVhcrKrVldU1OJfh8MMPxxFHHIFDDj0UgGBleQVFUWJhcRGLCz10Ozm+cMcX8cpXvgp/+c6/QFlipgPTlJ5MZTZGO76BjWe9HPmmzYDL0H3c4eBgjOXv3gY3i4LQxxtEFIuy8OVbz2wJ27FaWolShlCYbuIjWdZjMndgkPiHCyElR4dghje8IZHmMYyZhRgyjWAbUyvPppK6zGE4WAXgcO5Pn4vLXnIZLrjgWTjyqK1YWlzCZH699qdiUv8Qt3/uc7jhQzfg1ts+g7IYottbrBTzSfMBqRoH5Qi9Y85DtvlIOAAiHQz3PoK1732mygMbrqP9/LBeLrTS3dHuNebMlvUdC6FGdH0BikmHYtpmi2gvepgS2w7VWpND1y30pJPXUUVDEBiNVgEACwsbcMwxR2P79mOw7eht2LhhA0hi//5l7Ni5Aw987wHcd9/92L9/DwBMquYMRTE2zyG5TYkDR2tRRufyfiBe3iSbxnXK6HIdC5qGEOUBKVMqIuVW/mDqONNQUTD0YdTSngmJ35QuCxqVV2PeRbNMbTbVix6NUJaj9E1zOTqd7ozwylaLLeVJ7GZEiXmuXbRQmG3Qd2bK+0sM9bE2YvFNi715B/Dk2bo0P6gBwwk7EGx6myJn9LZ5CpujgKpB3KRVbCCJzs24g5GkDcua5G5jtGl9bcZCqsmicJ3wVkogMuEooBaa0jriiT9MJoJU26CuEZ2sLpvehBbfs/Sk2eDI3jrxbfHADTFtU11UexiGKpU0MOZgsuOkYSEaRRcl9qgT0YXLW++WbLDWwAGomxkaYzJVSKUY8z4eA5piSFbQYC2kgFOo7GZ71td4MBGRhoY1qeWDZ40HxDMwpHWttIYQjKlKBp3vEMRNdDEsl86IuBrr6ogcoJ5BkzejYU8bawLS+39vanEW4sPuhSBBKvV7hAkU3DQfhK60Kg0WrBqtY5aGUmHtKIbSBpukSZRDxJ6PoTUUqY7vSqJrC5W02UyUDUgBon0OlSF/azhTyTMlpivEjE8qpJy0JLOL2zIK34zxgUhFpTQakGZCCybhS0vLG0TT6fM6ISEeFi5mSWNcktxa/MJLYXmQRkZIU+Yt9TPUOk6WnWvLurs2I0JZh4Mm/8MdE4nqi0mQyvIu514boqjTr0fhdJ5j1WfJjXxQ2xu0ijjIacSb/p9iZXNZW9uzxDlX+25sAhPjhKFscoX1xedYrX3ndUCqc3MiE4Z0SkFWQwDcnITAssZIDh0n5xX0pBhMWq6tp1qNi6m2xLC2ubxi1dXOBlQSKXJjRUZN1kEzPdE8yLzZXXEYjwZVSyfvNFZpxXhQ+3vlZFSCZHydZMXPiyCYDjLnousej4cAOHE3KiBwGBcjgEU1eBTeJwnIzzVsMAPGq/N3NF/wMEHfnKeaxuN4tTpQ3ksbB6kvQMvqJEhfZLatS4vFmChO1AXIhGXrAXxI24tqNsab/92JQ1GOsPWorSAEDz/0IFyWTfRgAmCdRJY5nHHG6Tjo4ENQFmM8+ugOfO1r90wWh8yijEzSizx3eMpTnoLNmzfNhc7LAv/+73dheXkZTtyskiuLMbZu24aNG5bwzW/+v+h2FzAuxti8aROOOWY7vvb1b8x+XlO1YrD4ZLQKbtqOzpFPRbnr2xjv+AZcZyGI8pPfGA/gDjoOeb6A0WPfBsRB1S0M9ZxbdSsOBGhms6FlrRcs+VUJid/EIJBBqZIW2FbEkECgdCCJXu9kK3KCYjzELR/7GH7uOc/B3//99cgyn/40H5ccj0fYsmULbrnlJjzu8MNx0kkn4fLLL8fZZz8V//LJT05IONXnVj8/wBFHHIFbbrkZW7YcgZNPPglnnHE6Tj31NHzxi/+GfXt3I8tyT2mrxOGHHYaPfORGfOELX8QjjzwMAfCRj9yIPXv24it3fRl53pl7f6gcPkAkB0eryE+4FP2f+UvA5chO/jW47kaUD98BybpeyZMBxRDZU34T/fP/BJ2Djsfadz5csWjIuO8uiPvHpvHkenk0hOlS38CayrW2F4MfjBkSRCuj66Sro7K4kg6Q88+u9FfWcNFFz8MPvv8A+v1FnHf+s/D5z92OTqeHooxJ53newX33fReXXfaS2d9vve02XHTxRbj5po+i21tAURSza19c2oz7778Pl132i9EW7Fw+c04CiTzv4t57v4PXvu5/4X3v+wecfsYZePOb3owHH3oI73nP36DbXZjkgeHMRSAZVwyRbToK3f/yJqx97HKUu74G9g/F0qWfx/ihz6Hc+TVI3p8IqZeQTh+8/0as7vkheideiswHOkQMoxvGkapVsyFV5kjLn4l97nKVeh5sh2ywaYitu6x5XosxESz4ms1HLH5ZBRLiVb9+Bd7+x3+KQw45BFdccQU+d/tnqy2IY08udtLzzab8xurPeDzC8v79WFhYqFXN01Mfj4dYWFjEaaedBsDBOYddu3bioYcfrvKuWm45Rre3iFtuvgmnnnIKPv2pT+HRHY/iZb/6sonhIBspTwKHshwDjzsH5Y+/gXLX1yALB4OruzD47i1w25+LcsdXq8+ebMUkwf0PonPoqcjETWy2pwVuQPqo4a1I+L80RDkyurf+mvGfFxsck6oIaHUkQjdMVQtOvBymjaxsk0OSD/EgNmeZbqmjAc4++6dQFiW+8IXbIZLht3/7N3HyyU/Gt771/1V2B2RtkxsOKquG333DG7C2uoqnP/2/YDwe4+abbvZcKecLcbi2gi1bjsAb33gVxuMRev0+PvGJT+Ldf/M3cJ0OirIIIkhVFF177XX4gz+4Br/+338DKyvLlevScBhXseImK6WsLUd2D8V4348BySEkSsmA1YfRPejYSsOBjOU2XI5xyQAWV6oeoqajqFe0DTuQSHJRsVXknP9OXgcztXlbCaJjqmcsLU0JdTkHtMg9xVUR4BVXvALHHXccrrvuOogItm/fjssvvxxXXvkGZFmO0XhYyy3LskQxLqrdC4IP/9ON+Kcbb0RZFHOFeg9+6C8u4YHvfw+XXvrC2qVmeX+y+OZE28p7eAznHD74wQ/gV37lV/DLv/TLuOfue3Dnnf828/2tvcKjtckT6M2ORQDZykPonvDzGHAMFANkLIBDnozRrq/HQ4+TF3Q0GqAv1cMsfKV6IpjpUfjCB1BnyLpgGE08E/4CTOz/pmfYPBxTpCGHYPP3RJMlU/yCRTAaruHUU0/HKU9+El546YsAVAaAS4uLuP766/Dud78HDz70w0lBMj1eRSJdWdmHP/zDazztwK63+DgZQ5jf5sXFRRx++BaMRqOJkDixb/9+lKVE8nTFaIT3vOdvcffdd+P666/Hgw8+hHe96y9x8cXPxc6duzysToByjOywU1FkPXDHPRDXqVw08y7Khz6DwRm/g94pL8Pom+9Fdsxz0T/iHOz/1zdAshyCctbTkBmctAhkXRTTYqpksMUy3qH8DphSGKXMqGkiFE3MmPhncn3oOOgzRuoD0oJFK4HenihJK+vWWxbCOPm2cw5jlnjOz12I97//A7jv3m/PxCeBEp/61KdxwQXn4/rrr4PrdFEUnJ326soKHnjg+9iwYTMGg0GljlAUc52T2vyIw769ezAej3HttX9bqeu7DE6AV7/6f+CBB76HbKKsX+nCrOG8Z56PQw6tctF+fwm33XYrPvzhG/HKV74KV1/9xgnmWFQYXzkGjjwf2eZjMP7xXUDWq7QCJUM5XgY/ewW6z/wTLJ7wYpTjHGu3vRwy2AlkvSr/q60XQbn2GPL9P7Sn5zTRIhHT5yS2VpAWjOk2qVccO71OSAhWIo7VloOPNbeANnSulLO6qACoc4LhcIBOt19rgY1HA/QXFiszQr/dLwKWxaRrMhEvV1na8890LoMI0e31Jl87FMUQg8FwPmw07Q6ITOZK9iPLu1Wl7jIMh6uT8yniRhurnyknC2q2CJwDxmsoCeSbjwb3P4KiGCHL+yBKTzzc6/OWZRUdWU4k16BKlTQLzx8IqXe9bk713HPCiG7B/2OThVTIqm4iRra8CGUBlhOV00hwTByKopipkobv4LQaFTFyz8DmlQg9gss5/jepQqcYYiV+iUmVynnBVIyrBaxCWai1FecephNAZTyosD/JwHKsC4MTik1tSM2ien36uG07TZx2PEedqe0PolWM6NDiyhTANnIztQXX5m0z5jqY0mfxHho1jZmpW9FEbUpkpoIvtYGgqXrWtJfK6AWUWdHDmc3XvO9c/z1xFa9j1omZXIuT+bky3ClmtZvUAfTpZzkHFhPMb2LnwFnERIImH3TsGiAx8fM6hrMe+sJpzSdMSLvLfAs2rJcYypa1IKxaTWsTbadqQR+PMTNNqfIwqLIskWUOLImSJTKXea8FUYyHAByyvFNR6llWRcbkNpcksiyr+sGTajrvVNVqUYyrll9RVItumoeVBVgWcFk+66pUkbFEWYwAccgmrcLZi4DJ77FAlnUmC5GT9l4xc1aa5qvk9PhufW3NWeNZYvtVqKvUPva6qmY9TYsNq0V0fWdpg91RtZeK6tmU8CFjU2ShUdRIKIhID4MinBNsWFqqyI7OYWGhP+/ZsoRzGY459gRs2XIESGLDxg04evsT0O32JosY2LBhA8piDLLE1q3bsXXbMRPTQ2JpwyIAYNPmg+Ac0OtVc8HdTgfHHns8ut3efBFNhtq3H3McDjroYJDEpo0bkbkqZyQLbNi4EduPPQHiMpAFFvp9iAg2bNyIPM+wuLQBALB58yZs27YdnRnG6fMfQ/OesL1JRaFW6jp/qpmQwliSdXiphIUMY5E3Z5qNNLJjQipH86shlhWA0OgJKwvNqL7cpDd8/PEn4LxnXYBiPESn28MTTz51okxVLchiPKiGzA85DGUxxPEnnIgzzzoT27cfi7Ic46it23DBz1wAcRm63T62btuGQw45GI87/DCIE5x62ukoxgOceOIJeNYFz8ZR245CMR7ilFNPx6aDNuHggw+ZReOyGGPLliNw2lNOQ6fTQ1mMcdbZT8U5P30eHrfl8XBZjpOe+ERs3rwRGzZuQrfbx/EnnoDxaA2nnHIqfvbCi7B161EoxgM8+ZTTcNTWrShZeuY/QYUrmvQuY81pq4CtOZdKS96VoNFkI6TeeafgIpo5NbtWg44vCgNZU78ide5tUok/YZ6tIPJlSXR7Czj0sMOwvLyCw7cchdFohNFoMOmdzvl6jzzyMH6yaxcAYDgYYu++fZWiveQ46qhtWFsb4sgjj8JwsFrJu3W6KClwLsfaWgUg79q1C/v378XBhxw20Z8usbC4hE2bNs1A56ld7GBtgLIskXcW8OCDDwIgFheXwLJEWZRYWFjAwkIXo9FgQv3P8OiPf4zHHts5O35RFBiNRgm+LNNKs9IWPp7vRqIKQWmyvDQNFKk5Pnkdkzkdi01ewKFNvZXlJnhnEfE1RTxN5RH6uU3zuNFwiIXFJQwHa8jzvLJC9RLsTqeDohijKKrcq2SJPMswGg3R6y1gdWUFvX4H41GBgw4+BATwk1070el04JxgPBqj0+1iOFhDt9evHC9djkMOORi7du2aFxQskXe62LRpM1aW92FtbVDlgSTyLMdwuIZefxELi33seWz3hE/Yx3g8ruy+RgN08hzjcYHFDRvQ7/Wwe/djKIuyTjYQm7wbap41zncnoKlWw+YmGdYaTM97jIkGNNSwtMQVKWkEXXRSFLYMxRg/bFd1zZjAk7HKeetrUs1OcsQ5c3guezbND6fdikq2rag4h8V40jXJZ9DMFGqRSfU6JQiURYkszyL4qpgUFG5SVPjFBsuyKpryfA7tzJ539f2pjBzLqm2oF5myTs5fvDjqghb8D+CC69CnqRYgg8k0/7zaDKxri7a5HG8158qU+6IYo5Mppg7iiDo559Q7m+ybqk4Bk88mIU4UoVEN8kjT5FXZXd/7jXMYCoHOo3hw0CyAELpgQKRB2GKBaUz25MC7vwDbIt3UiKQapZ7tVKyszksytFsaMoYSQ/LloImpSejUKQ3ztbRkyGCoRcCAP5jmTFJrrwUtOO2FDDtTUSbE9Ba8XnZ7y0jo/kNtFko8XSXGNJnmNdaoyM+aR7CYooqWgQwNF0d7Ik9m86us6xbO5lACeyqGVgbeQqWekyUdSrXFR8NHjqHiqeI+4L9k4nkGi1YVa9YO0q7+9Z9B0uxGQr/gBlNqiqKeaRkVejkixbs2Y1hdpEWHZF79slGDIHBmMvMeg5EpiNjg0eRrSHcXgz4m1CWJqbyEEfRmuJrXhMFhwFcahKIxo8XQ+mkPurA1Z9A+lkt/RFCEMKEgUBOjZN1Y0HTn4TpmTptmdn2LCBoPVNLOR/7PU3P/UY6jLRiGcBaUNqcSCYXxXLCXHrDRxtZSpJAG1ydRdo3WAgnG45BWYguuEWzWbBk4l9Cfb1Va8jqPhqJZzIvv/mi5kXMdXrSINVNkHfRJ8aONNFSQEssW1xjljI22GeoeSGxZIZ5ON2k8dbZ8YSURUFI/L9C8S9sEg7DhJQ1+V25djojWCwjEbaEaQh+qn9J0EFNxLST6yFrLLpXcU9CkciIRqB6or1IULRzWtzYirixFguCnePNxao/gi8TPxx/s3rxhsZXcMYwXuLVTpsV4ZuPCnT4212Qsp42ni2rXGWx9qaBmyv4DwtgKNbJMpWKgbfUpI70jpk6sbtIiTBv0qTanjEcc/Acr1ssgtXkNHSlgS8UyKJ5zbOg4sd0iXVdLrkF+hFEOSPPmUhW0UbhplHUMDSjyNiFT16sua85NlISbtxhJd8rDInHDfIPo2fX5xtGIYZ0abCOK3azXA/ePpY0vRufOuMMUkULaZHFUHIxE1b1pbsfFi5zrLkKotd3YHH2hDUKLEdrZzn5BidcMjfqERjGk2dUzPg/RYCUtz/Dug4SRlGl7VdGcN1lfcMIWaVyI+4lODLF2DdVArl6ViyVll6pqRUw/ZTF3ibAIqVnAe2+QPwJIps2sa56HYR4ksYtmCzft6KEmqznFAYgBFigNtvPiQxtGHsHUoL2h41dLB4KIFTT0JWXKo75Y3mioxnihkhfScBAQaZD/heE/LM1MKoPU4FTRRQb2XGEHgmJDMmIwdOFJqKnFBE2Jw/pbyxZeZFSM/poWvigRRjv/UBpOoy21dSHyjyGG3aIGThuAb3R/DWxVNMiEiUk32vzAoPWoM3Jo5v1OE1CkRgSo5WY0nLwVT7eIzKBx1KRGZG3ais3FwwDRZ8vBG60ylkRXmIyLCoFCCE36pQazu+vJmSWouhU8gUGbk6J7LAtMV0sklU/j7zBCDmhf/mxuRjmcwOrA+3K9YkAaSKqm1gaMUgUAYRiz2IWMhNFJtG23foNswk3i2jQPZNBoK2qRIbQys3LkVAWr5JIRMqW1HBuAf9Lo3OiNBZrXIw2dkZljuu7tW6P1RAxmP89gw/am+cZKQFClAZhq7Bkbp2ISdRL1fKgqjzZBHdJQRNFDFEWB1wjxbU7VwXAYptEN+4TQIDQErT9VC1zxgLbYUOHzbti6xAC1XVJ60pfhJQ8MC2KLGVSmmLYh8bJFIaLOq0CHYdQcNxUFtByMpgAvYTHRPNtm0eEOtpHClQR4LEq7UCSWaxMkCgUGbCjoxYoofe4WaIcz3+yQqRLRnaSu5Zxs31muSIZ6vaQa5Paon969MT4jmoFm0LWJBRMlVXFb2yipRzSRhDKslpIkjL2hUOKYoklJDI/Ro59pfWxqkBINKUhpVMqW2gKkUm1qA8xqQZLisCHNBBGpzSCYIuSWC5Cp2R4O14itkaLcMEvDnq1zXd2rgak8TxJkjZBPKKmckQHcAnsnkHqwkVqKpfTVqanz62bh0rKR52pvstjOjJK8aZKGC6w3MQkH0p4b8XMd0rRlELFzN0lQomjZNZAtuW5N/eZEZNY6OD6JlJb1RYB3+tJxbMLplAlECckWDUbXSpyHTqyrHcHpTWgaxipNeVcCGhFFCFNC1maYLCufJ0zoCQfeGIm8lRGHzkB0IthCkvEwnbGJ8Y7RK1mQNqq2ptR8sL+R9QKdIZMafWCLgoN2razZ58iMkCpGwk1JIPxacUIdfVdBUdbFE825YBigtyhvrrINTVUTmHpN0mYqYi0eKv9Eot3odti2FG+TNpZyrYshehEkSo4oTBRU2q0TpUpG8+8KDDBdFOekiA+YauFYdB0xSIwMGvhMp6KiIfI+7NMgBaZ6v3kdmwiyZHw7ok4Ma0pWatWnEXHMHnjD1ww3fgMXFehyeaTS9qIxt5EA5QX6tus/B2FzV8Z8CUNKPo1dWhTwMulwFOSNqmma2FWUiK2cGuF3DN6zuI3EWusr5QfHWiHERmxLj040iQliFwHUpCuMcwhNRSgJwoA0vKiWukWbQtPoVUsbSlb8784u/RvyhhpzJh4DTFN5ROfHhdmSGALq0Np20sBr88ieqWqdEqP6ggPoJzd6lCvtTsWgUbxUKKqEUy9HE/HU28FocCQJo0fcfO2SqtC9bdnpLpj2oJBoHDvRIBzNeoHGzYon4tgW9GabGY/6A2Zqyk8apMbYZFJQ5/LRnIFJDPaH/EFF4VSUofL4HJWcj+EWzdi+wd9qKS3IClb7V+Kef7BfuDr0It47q2uN0ELKoz6otbhoY1hsyQKJ2o/SwL5hu15nOFYZ9Vt1A2amKEisj5bWCRMa20TSmjg+9Z0M2ppiLFgE5NlwepA6B8Bk/dNmtgsaTNwknAvWwwUN8RkTeCGNMUBD/FyrikVjOdPYXuZ5Ub2f2zD5Fm45kqrolEGh4ObThKYM61QRIycTJZmXNOZqnpc0q+4qwlBUh8poj3IytdWGaIRYfEBJ+H3pohFR7sWmvETJOyzoQ5RtBNasiZGnURJYmsQFD1PqntpIKhO/H0ZasWn+Wvog0l59ngoLSFuw1DDXRAtRvC2elmm4zKwlJEnOTYPuzj44GrwxLYBWW64pWjcTXsXBTISkbKWZngYjbUKRWaVL4mtRtG0k1mJWB5qYpEdJS3vcebeULdJli6zhLyGJ+8ICvVBpPb+tcw3nvWBpa9OU0AlMGdc1Svo3edrS/L5mgBWRUoNFRqRwyQS1TNDQhUkk5iKGG7w0dGnSx2b0TA6EsaRsoqKoXoiYAgVcN4E4lOhlE4dPG8VsM/fOFovKKjTY0MyijolJiwUSFQPxvINE0hVssCgNmOCN7zXRSIdmSwqbSAvhjLAZ1rTIRRnt1AgICbitRUxzdlURl8xt2lbN3nBGQWFBMwFBkg2u6JIST7cU/CWmgrEV8dTgFLLNHRFjwEdpJbZ5koTa/hIzL5PElCW922YPUMWjCevYnEXNAdHAOG5G6MSUsdE070QfpdQF/e2UINJUaSuYEzo2SWKEsUXHwNRrMe5uZHOGdsPj1KbbZB0aCDQ3RlFvmwTdGjG4nuuV8og6IQiwQDSApm0aL2zMA9I0H2mMzskHRk1DWYKGggTDO1D0blKD3WzIV9cjhZHAScPF1prxYktoRKmVSJq0IPYguiR3Pf21cLrqVdzaP/A/Ys4GtLqHDULYtVS81tD3J/NoVHLK1iJsyetDAnsUvZQIMU6hd29FSdyl9Wve/klIQ1lDO7I3iIiyUaWC0cC8ay1McMA3IS3XQLY9tLUYvCVd86JIsJVJT90Lirs4212XNFd8bFBJYHJLh93BSY0BGHBZWgZYE49qGvJigwuWNBQwog0lpVYHkzdbWiUhrNsWi7IdSUtZCI2fKKnpfNHZOiqNSlo8AKOroTiky7qddltAPgw1ZFK4raiccJOkoOjriyI0Kkl4zcJizaGkBLetyf5TRQ2o4EfGxEXL2dwkACSp4kfpWES92NSMSjslKGnHXU10WdYB8obyHLDudYoka+joqFliu1LfTFwkHs50GjrdzOkVVSdE1L4lY1/a1FvK+uVKdLuYgOJEZ4RYsIZY2CWUrYgN1aQoLBDrObEZV0NyXTREHDG8/9gy/kpSDSG0otEBchpnWL+/To9IbGBoQG85JRlxko6qxvwRqV+GaJo0Fpkg1ChhQsWfdSVEMWeOmySImWDWS3vTKwYTfzIXUheTVEpFtaJe+TfOqSDR2ampPaQnYeJXWJS5YDbNMRnkakljP9IU8q2pfjYRL6kA46LLj1k0LSuh1uh/NdH2BHDc5LdHKouWLWovDYcTczxYRBobLG3I8/75SjR4zwMUtGRQhFBLLCWhhCCJroTyGMjm5D6yAGnnT8bk0JOWIEsCrwu+L9aCsGAYaQCSJeE0yXVhKWwgK5Bstmu2qL8iKhQTzTlFmrnSMnPVIqDGBZQUUzqkt9t9SqmJEdkSvTV1ztQcqyhClAwNq5vGyjWaF5vm4OK8TjTQm0kVUrQe5uSBwn0671EZa1Wbl2ynid80aiWtoqsmUk7rZCQO82YOxASpv0VXQFLZUtgGEmXttzUVkGDuoi4ZQo1po4LaoW+KpIYxdQKGdbxG29zUdWmUMEHKSsj2AGyDNTa9WqL0gkPauhiC2uuBoUVSmaNBdxJfS6qhs8gWegRMVImMo580mGuHJjAiDYC5qNGUKv9REp2HA/VPCQwhRBK8vmaIKTXpawm91ykkevfLNRvFKHUgCaxjM9HffFHnqQm0DujTc9GvQGxxn1aUMRtqidM6aXQ7ZiMXcn2lgi2CoZdrNPUI0UIFRzzjV2XfE2kUr9RgvmAqThrJj00dXIvwJGZDpN3iTy5IadNj0KjuksDgLFFG2maxcqCxqo3JDG3MOH2XE5Sy1EstSTSDLa8uzaev/vX/B0Y7VzOLzoReAAAAAElFTkSuQmCC" alt="Apex .01" style="height:72px;width:72px;object-fit:contain;flex-shrink:0">
        <div>
          <h1>Apex .01</h1>
          <p class="subtitle">Operational dashboard for status, risk gates, execution, audit trail, and visual analytics</p>
        </div>
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
        <button id="subtabReasoningBtn" class="subtabbtn" onclick="showAnalyticsSubtab('reasoning')">Reasoning Patterns</button>
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

        <div id="analyticsReasoningPanel" class="analytics-panel span-2">
          <div class="kpi-section">
            <div class="kpi-section-label">Reasoning Patterns</div>
            <div class="kpis">
              <div class="card"><div class="kpi-label">Trades Analysed</div><div id="raAnalyzed" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Overall Win Rate</div><div id="raWinRate" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Phrases Found</div><div id="raPhraseCount" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Generated</div><div id="raGenAt" class="kpi-val">-</div></div>
            </div>
          </div>
          <div class="card">
            <h3>Top Predictive Phrases &mdash; Wins <span style="font-weight:400;font-size:11px;color:#9ca3af">(positive lift vs baseline win rate &bull; p&lt;0.05 in bold)</span></h3>
            <div class="table-wrap"><table id="raWinTable"></table></div>
          </div>
          <div class="card">
            <h3>Top Predictive Phrases &mdash; Losses <span style="font-weight:400;font-size:11px;color:#9ca3af">(negative lift &bull; p&lt;0.05 in bold)</span></h3>
            <div class="table-wrap"><table id="raLossTable"></table></div>
          </div>
          <div class="card span-2" style="font-size:11px;color:#6b7280;line-height:1.6">
            <strong>Methodology:</strong> Unigrams (&ge;3 chars, non-stopword) and bigrams are extracted from each LLM reasoning string.
            Each phrase is tested for association with win/loss outcome using a 2&times;2 chi-squared test (Yates correction, 1 df).
            <em>Lift</em> = (phrase win rate &minus; overall win rate) / overall win rate.
            Min count = 5. Results are reproducible from raw trade data.
            <button class="research" style="margin-left:12px" onclick="runResearch('/api/research/reasoning-analysis?save=true', 'Reasoning report saved')">Save Report</button>
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
            <h3 style="margin-top:20px;border-bottom:0;padding-bottom:0">LLM Tier Comparison <span style="font-weight:400;font-size:11px;color:#9ca3af">primary-only vs secondary-escalated decisions</span></h3>
            <div class="kpis" style="margin-top:10px">
              <div class="card"><div class="kpi-label">Escalation Rate</div><div id="tcEscRate" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Primary Accuracy</div><div id="tcPrimAcc" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Secondary Accuracy</div><div id="tcSecAcc" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Primary Signed bps</div><div id="tcPrimBps" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Secondary Signed bps</div><div id="tcSecBps" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Accuracy Delta</div><div id="tcAccDelta" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">bps Delta</div><div id="tcBpsDelta" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Primary $/correct</div><div id="tcPrimCostCorr" class="kpi-val">-</div></div>
              <div class="card"><div class="kpi-label">Secondary $/correct</div><div id="tcSecCostCorr" class="kpi-val">-</div></div>
            </div>
            <div class="table-wrap" style="margin-top:8px"><table id="tcReasonsTable"></table></div>
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
      if (!overview) refreshAll();
    }
    function showAnalyticsSubtab(tab) {
      const names = ["charts", "quality", "inventory", "patterns", "context", "reasoning", "symbolGate", "cc", "decisions", "trades", "events"];
      for (const name of names) {
        const panel = document.getElementById(`analytics${name[0].toUpperCase()}${name.slice(1)}Panel`);
        const btn = document.getElementById(`subtab${name[0].toUpperCase()}${name.slice(1)}Btn`);
        const active = name === tab;
        if (panel) panel.className = active ? "analytics-panel active span-2" : "analytics-panel span-2";
        if (btn) btn.className = active ? "subtabbtn active" : "subtabbtn";
      }
      refreshAll();
    }
    function analyticsTabActive() {
      const panel = document.getElementById("panelAnalytics");
      return Boolean(panel && String(panel.className || "").includes("active"));
    }
    function currentAnalyticsSubtab() {
      const names = ["charts", "quality", "inventory", "patterns", "context", "reasoning", "symbolGate", "cc", "decisions", "trades", "events"];
      for (const name of names) {
        const btn = document.getElementById(`subtab${name[0].toUpperCase()}${name.slice(1)}Btn`);
        if (btn && String(btn.className || "").includes("active")) return name;
      }
      return "charts";
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
      const padX = 32;
      const padY = 20;
      const dx = Math.max(1, width - padX * 2);
      const dy = Math.max(1, height - padY * 2);
      const yLo = (minY === maxY) ? (minY - 1) : minY;
      const yHi = (minY === maxY) ? (maxY + 1) : maxY;
      const toX = (i) => padX + ((i / Math.max(1, points.length - 1)) * dx);
      const toY = (v) => padY + ((yHi - v) / Math.max(1e-9, (yHi - yLo))) * dy;
      const poly = points.map((p, i) => `${toX(i)},${toY(p.y)}`).join(" ");
      const baseY = Math.min(padY + dy, Math.max(padY, toY(0)));
      const finalEq = points[points.length - 1].y;
      const lineColor = finalEq >= 0 ? "#16a34a" : "#dc2626";
      const gradRGB = finalEq >= 0 ? "22,163,74" : "220,38,38";
      const areaClose = `${toX(points.length - 1)},${baseY} ${toX(0)},${baseY}`;
      const gridLines = [0.25, 0.5, 0.75].map((f) => {
        const gy = padY + f * dy;
        return `<line x1="${padX}" y1="${gy}" x2="${width - padX}" y2="${gy}" stroke="#e1e4ea" stroke-width="1"/>`;
      }).join("");
      const svg = `
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <defs>
            <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="rgba(${gradRGB},0.20)"/>
              <stop offset="100%" stop-color="rgba(${gradRGB},0.01)"/>
            </linearGradient>
          </defs>
          ${gridLines}
          <polygon points="${poly} ${areaClose}" fill="url(#eqGrad)"/>
          <line x1="${padX}" y1="${baseY}" x2="${width - padX}" y2="${baseY}" stroke="#c4cad4" stroke-width="1" stroke-dasharray="4,3"/>
          <polyline points="${poly}" fill="none" stroke="${lineColor}" stroke-width="1.5"/>
          <text x="${padX}" y="13" font-size="10" fill="#6b7280">max ${n(yHi, 2)}</text>
          <text x="${padX}" y="${height - 4}" font-size="10" fill="#6b7280">min ${n(yLo, 2)}</text>
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
      const pad = 24;
      const plotW = width - pad * 2;
      const plotH = height - pad * 2;
      const barW = plotW / bins;
      const gridLines = [0.25, 0.5, 0.75].map((f) => {
        const gy = pad + (1 - f) * plotH;
        return `<line x1="${pad}" y1="${gy}" x2="${width - pad}" y2="${gy}" stroke="#e1e4ea" stroke-width="1"/>`;
      }).join("");
      const rects = counts.map((c, i) => {
        const h = (c / maxC) * plotH;
        const x = pad + (i * barW) + 1;
        const y = pad + (plotH - h);
        const binCenter = minV + ((i + 0.5) / bins) * span;
        const barColor = binCenter >= 0 ? "#16a34a" : "#dc2626";
        return `<rect x="${x}" y="${y}" width="${Math.max(1, barW - 2)}" height="${h}" fill="${barColor}" opacity="0.72"/>`;
      }).join("");
      const svg = `
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          ${gridLines}
          <line x1="${pad}" y1="${pad + plotH}" x2="${width - pad}" y2="${pad + plotH}" stroke="#c4cad4" stroke-width="1"/>
          ${rects}
          <text x="${pad}" y="14" font-size="10" fill="#6b7280">min ${n(minV, 2)}</text>
          <text x="${width - pad - 72}" y="14" font-size="10" fill="#6b7280">max ${n(maxV, 2)}</text>
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
      const barW = 68;
      const gap = 40;
      const x0 = 68;
      const baseline = 142;
      const maxVal = Math.max(trade, hold, longDir, shortDir, 1);
      const scale = 96 / maxVal;
      const gridLines = [0.33, 0.66, 1.0].map((f) => {
        const gy = baseline - f * 96;
        return `<line x1="${x0 - 8}" y1="${gy}" x2="${width - 28}" y2="${gy}" stroke="#e1e4ea" stroke-width="1"/>`;
      }).join("");
      const bars = [
        { label: "Trade", value: trade, pct: Math.round(trade / total * 100), color: "#1a56db" },
        { label: "Hold",  value: hold,  pct: Math.round(hold / total * 100),  color: "#6b7280" },
        { label: "Long",  value: longDir, pct: longDir ? Math.round(longDir / Math.max(1, trade) * 100) : 0, color: "#16a34a" },
        { label: "Short", value: shortDir, pct: shortDir ? Math.round(shortDir / Math.max(1, trade) * 100) : 0, color: "#dc2626" },
      ];
      const rects = bars.map((b, i) => {
        const h = Math.max(2, b.value * scale);
        const x = x0 + i * (barW + gap);
        const y = baseline - h;
        return `
          <rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="${b.color}" opacity="0.80"/>
          <text x="${x + barW / 2}" y="${baseline + 13}" text-anchor="middle" font-size="10" fill="#6b7280">${b.label}</text>
          <text x="${x + barW / 2}" y="${y - 5}" text-anchor="middle" font-size="10" font-weight="600" fill="#1e293b">${b.value}</text>
          <text x="${x + barW / 2}" y="${y - 16}" text-anchor="middle" font-size="9" fill="#9ca3af">${b.pct}%</text>
        `;
      }).join("");
      const svg = `
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          ${gridLines}
          <line x1="${x0 - 8}" y1="${baseline}" x2="${width - 28}" y2="${baseline}" stroke="#c4cad4" stroke-width="1"/>
          ${rects}
          <text x="14" y="13" font-size="10" fill="#6b7280">n=${rows.length}</text>
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
      const el = document.getElementById("qualityChart");
      if (!el) return;
      const width = Math.max(520, Math.min(900, Math.floor(el.clientWidth || 520)));
      const height = 260;
      const plotTop = 24;
      const plotBottom = 166;
      const baseline = Math.round((plotTop + plotBottom) / 2);
      const labelY = 202;
      const metaY = 222;
      const amplitude = Math.max(24, Math.floor(Math.min(baseline - plotTop, plotBottom - baseline) - 6));
      const maxAbs = Math.max(1, ...entries.map((h) => Math.abs(h.signed)));
      const slot = Math.max(110, Math.floor((width - 84) / Math.max(1, entries.length)));
      const barW = Math.max(42, Math.min(96, Math.floor(slot * 0.62)));
      const gap = Math.max(26, slot - barW);
      const x0 = Math.max(34, Math.floor((width - ((barW + gap) * entries.length - gap)) / 2));
      const bars = entries.map((h, i) => {
        const mag = Math.abs(h.signed) / maxAbs * amplitude;
        const x = x0 + i * (barW + gap);
        const y = h.signed >= 0 ? baseline - mag : baseline;
        const color = h.signed >= 0 ? "#16a34a" : "#dc2626";
        const valueY = h.signed >= 0 ? Math.max(plotTop + 12, y - 8) : Math.min(plotBottom - 10, y + mag + 16);
        return `
          <rect x="${x}" y="${y}" width="${barW}" height="${Math.max(2, mag)}" fill="${color}" opacity="0.78"/>
          <text x="${x + barW / 2}" y="164" text-anchor="middle" font-size="10" fill="#6b7280">${h.horizon}m</text>
          <text x="${x + barW / 2}" y="${h.signed >= 0 ? y - 5 : y + mag + 12}" text-anchor="middle" font-size="10" font-weight="600" fill="#1e293b">${n(h.signed, 1)}</text>
          <text x="${x + barW / 2}" y="176" text-anchor="middle" font-size="9" fill="#9ca3af">n=${h.count} ${pct(h.accuracy)}</text>
        `;
      }).join("");
      const gridLines = [-1, -0.5, 0.5, 1].map((f) => {
        const gy = baseline - f * 62;
        return `<line x1="22" y1="${gy}" x2="${width - 22}" y2="${gy}" stroke="#e1e4ea" stroke-width="1"/>`;
      }).join("");
      if (el) {
        el.innerHTML = `
          <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
            ${gridLines}
            <line x1="22" y1="${baseline}" x2="${width - 22}" y2="${baseline}" stroke="#c4cad4" stroke-width="1" stroke-dasharray="4,3"/>
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
        summaryEl.textContent = `${n(fullyCovered, 0)} / ${n(items.length, 0)} features have at least 95% populated coverage in the scanned rows. Unknown-heavy fields usually indicate older pre-feature samples.`;
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
    function renderReasoningAnalysis(report) {
      const ra = (report || {}).reasoning_analysis || {};
      if (!ra.ok) return;
      document.getElementById("raAnalyzed").textContent = n(ra.analyzed_count, 0) + " / " + n(ra.total_count, 0);
      document.getElementById("raWinRate").textContent = pct(ra.overall_win_rate);
      document.getElementById("raPhraseCount").textContent = n(ra.phrase_count, 0);
      document.getElementById("raGenAt").textContent = fmtTs(ra.generated_at);
      const phrases = ra.phrases || [];
      const pos = phrases.filter((p) => p.lift > 0).slice(0, 20);
      const neg = phrases.filter((p) => p.lift < 0).slice(0, 20);
      const hdr = `<thead><tr><th>Phrase</th><th>N</th><th>Wins</th><th>Win%</th><th>Lift</th><th>p-value</th></tr></thead>`;
      const mkRow = (p) => {
        const sig = p.p_value < 0.05;
        const liftColor = p.lift > 0 ? "#16a34a" : "#dc2626";
        return `<tr${sig ? ' style="font-weight:600"' : ""}>
          <td>${esc(p.phrase)}</td>
          <td>${p.count}</td>
          <td>${p.win_count}</td>
          <td>${pct(p.win_rate)}</td>
          <td style="color:${liftColor}">${p.lift > 0 ? "+" : ""}${(p.lift * 100).toFixed(1)}%</td>
          <td style="color:${sig ? "#374151" : "#9ca3af"}">${p.p_value.toFixed(3)}${sig ? " ✓" : ""}</td>
        </tr>`;
      };
      const winEl = document.getElementById("raWinTable");
      if (winEl) winEl.innerHTML = hdr + `<tbody>${pos.map(mkRow).join("") || '<tr><td colspan="6" class="small">No positive phrases yet</td></tr>'}</tbody>`;
      const lossEl = document.getElementById("raLossTable");
      if (lossEl) lossEl.innerHTML = hdr + `<tbody>${neg.map(mkRow).join("") || '<tr><td colspan="6" class="small">No negative phrases yet</td></tr>'}</tbody>`;
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

    function renderTierComparison(report) {
      const tc = (report || {}).tier_comparison || {};
      if (!tc.ok) return;
      const p = tc.primary || {};
      const s = tc.secondary || {};
      const accDelta = (s.accuracy || 0) - (p.accuracy || 0);
      const bpsDelta = (s.avg_signed_bps || 0) - (p.avg_signed_bps || 0);
      const deltaColor = (v) => v > 0 ? "#16a34a" : v < 0 ? "#dc2626" : "#6b7280";
      document.getElementById("tcEscRate").textContent = pct(tc.escalation_rate);
      document.getElementById("tcPrimAcc").textContent = `${pct(p.accuracy)} (n=${n(p.labeled_count,0)})`;
      document.getElementById("tcSecAcc").textContent = s.raw_count > 0 ? `${pct(s.accuracy)} (n=${n(s.labeled_count,0)})` : "—";
      document.getElementById("tcPrimBps").textContent = n(p.avg_signed_bps, 2);
      document.getElementById("tcSecBps").textContent = s.raw_count > 0 ? n(s.avg_signed_bps, 2) : "—";
      const accDeltaEl = document.getElementById("tcAccDelta");
      accDeltaEl.textContent = s.raw_count > 0 ? (accDelta >= 0 ? "+" : "") + (accDelta * 100).toFixed(1) + "pp" : "—";
      accDeltaEl.style.color = s.raw_count > 0 ? deltaColor(accDelta) : "";
      const bpsDeltaEl = document.getElementById("tcBpsDelta");
      bpsDeltaEl.textContent = s.raw_count > 0 ? (bpsDelta >= 0 ? "+" : "") + n(bpsDelta, 2) : "—";
      bpsDeltaEl.style.color = s.raw_count > 0 ? deltaColor(bpsDelta) : "";
      document.getElementById("tcPrimCostCorr").textContent = p.labeled_count > 0 ? "$" + n(p.cost_per_correct, 4) : "—";
      document.getElementById("tcSecCostCorr").textContent = s.raw_count > 0 && s.labeled_count > 0 ? "$" + n(s.cost_per_correct, 4) : "—";
      const reasons = tc.secondary_reasons || [];
      const rEl = document.getElementById("tcReasonsTable");
      if (rEl && reasons.length > 0) {
        const body = reasons.map((r) => `<tr><td>${esc(r.reason)}</td><td>${r.count}</td></tr>`).join("");
        rEl.innerHTML = `<thead><tr><th>Escalation Reason</th><th>Count</th></tr></thead><tbody>${body}</tbody>`;
      }
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
        const fastFirstLoad = !window.__apexHydrated;
        const analyticsActive = analyticsTabActive();
        const activeAnalyticsSubtab = currentAnalyticsSubtab();
        const needWideHistory = analyticsActive && activeAnalyticsSubtab === "charts";
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
          inventoryRes,
          monitorRes,
          symbolPerfRes,
          patternRes,
          contextRes,
          controlsRes,
          ccRes,
          ccShortRes,
          cbRes,
          autoHealthRes,
          guardsRes,
          causesRes,
          automationRes,
          readinessRes,
          reasoningRes,
          tierRes
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
          fetchJson("/api/audit?limit=20"),
          fetchJson("/api/notifications?limit=20"),
          fetchJson("/api/session-config"),
          fetchJson("/api/acceleration"),
          analyticsActive && activeAnalyticsSubtab === "quality"
            ? fetchJson("/api/research/prediction-quality?lookback=600&horizons=5,15,30&min_confidence=0&quality_mode=good_only")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "inventory"
            ? fetchJson("/api/research/feature-inventory?lookback=2000")
            : Promise.resolve({}),
          fetchJson("/api/model-monitoring"),
          analyticsActive && activeAnalyticsSubtab === "symbolGate"
            ? fetchJson("/api/research/symbol-performance?lookback=20000&horizon_minutes=15&quality_mode=good_only")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "patterns"
            ? fetchJson("/api/research/pattern-leaderboard?lookback=20000&horizons=15,30&quality_mode=good_only&min_labels=5&stress_bps=3&min_accuracy=0.5")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "context"
            ? fetchJson("/api/research/context-leaderboard?lookback=20000&horizons=15,30&quality_mode=good_only&min_labels=5&stress_bps=3&min_accuracy=0.5")
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
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "reasoning"
            ? fetchJson("/api/research/reasoning-analysis?lookback=5000&min_count=5&top_n=40")
            : Promise.resolve({}),
          analyticsActive && activeAnalyticsSubtab === "cc"
            ? fetchJson("/api/research/tier-comparison?lookback=5000&horizon_minutes=15&quality_mode=good_only")
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
        if (tierRes.tier_comparison) renderTierComparison(tierRes);
        if (cbRes.cell_leaderboard_bootstrap) renderCellBootstrap(cbRes);
        if (inventoryRes.feature_inventory) renderFeatureInventory(inventoryRes);
        if (patternRes.pattern_leaderboard) renderPatternLeaderboard(patternRes);
        if (contextRes.context_leaderboard) renderContextLeaderboard(contextRes);
        if (reasoningRes.reasoning_analysis) renderReasoningAnalysis(reasoningRes);

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


@app.get("/api/research/feature-inventory")
def api_feature_inventory(
    lookback: int = Query(default=2000, ge=1, le=100000),
) -> dict:
    return {"feature_inventory": service.feature_inventory_report(lookback=lookback)}


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


@app.get("/api/research/tier-comparison")
def api_tier_comparison(
    lookback: int = Query(default=5000, ge=1, le=50000),
    horizon_minutes: int = Query(default=15, ge=1, le=390),
    quality_mode: str = Query(default="good_only", pattern="^(all|good_only)$"),
) -> dict:
    return {
        "tier_comparison": service.tier_comparison_report(
            lookback=lookback,
            horizon_minutes=horizon_minutes,
            quality_mode=quality_mode,
        )
    }


@app.get("/api/research/reasoning-analysis")
def api_reasoning_analysis(
    lookback: int = Query(default=5000, ge=1, le=50000),
    min_count: int = Query(default=5, ge=1, le=500),
    top_n: int = Query(default=40, ge=5, le=200),
    save: bool = Query(default=False),
) -> dict:
    return {
        "reasoning_analysis": service.reasoning_analysis_report(
            lookback=lookback,
            min_count=min_count,
            top_n=top_n,
            save=save,
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
