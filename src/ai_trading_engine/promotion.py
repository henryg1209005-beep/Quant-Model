from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionPolicy:
    min_folds: int = 3
    min_model_selected_trades: int = 40
    min_expectancy: float = 0.0
    min_net_pnl_edge: float = 0.0
    require_recommendation_promote: bool = True


def evaluate_predictive_report(report: dict[str, Any], policy: PromotionPolicy) -> dict[str, Any]:
    folds = report.get("folds", []) if isinstance(report, dict) else []
    agg = report.get("aggregate", {}) if isinstance(report, dict) else {}
    ok_report = bool(report.get("ok", False)) if isinstance(report, dict) else False

    fold_count = int(len(folds))
    model_trades = float(agg.get("model_selected_trades_total", 0.0))
    model_exp = float(agg.get("model_selected_expectancy_weighted", 0.0))
    model_net = float(agg.get("model_selected_net_pnl_total", 0.0))
    base_net = float(agg.get("baseline_selected_net_pnl_total", 0.0))
    pnl_edge = model_net - base_net
    rec = str(agg.get("promotion_recommendation", ""))

    checks = [
        {
            "name": "report_ok",
            "actual": ok_report,
            "expected": True,
            "comparator": "==",
            "passed": ok_report is True,
        },
        {
            "name": "fold_count",
            "actual": fold_count,
            "expected": int(policy.min_folds),
            "comparator": ">=",
            "passed": fold_count >= int(policy.min_folds),
        },
        {
            "name": "model_selected_trades_total",
            "actual": model_trades,
            "expected": float(policy.min_model_selected_trades),
            "comparator": ">=",
            "passed": model_trades >= float(policy.min_model_selected_trades),
        },
        {
            "name": "model_selected_expectancy_weighted",
            "actual": model_exp,
            "expected": float(policy.min_expectancy),
            "comparator": ">=",
            "passed": model_exp >= float(policy.min_expectancy),
        },
        {
            "name": "model_net_pnl_edge_vs_baseline",
            "actual": pnl_edge,
            "expected": float(policy.min_net_pnl_edge),
            "comparator": ">=",
            "passed": pnl_edge >= float(policy.min_net_pnl_edge),
        },
        {
            "name": "promotion_recommendation",
            "actual": rec,
            "expected": "promote" if policy.require_recommendation_promote else "any",
            "comparator": "==",
            "passed": (rec == "promote") if policy.require_recommendation_promote else True,
        },
    ]
    passed = all(bool(c["passed"]) for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "summary": {
            "fold_count": fold_count,
            "model_selected_trades_total": model_trades,
            "model_selected_expectancy_weighted": model_exp,
            "model_selected_net_pnl_total": model_net,
            "baseline_selected_net_pnl_total": base_net,
            "model_net_pnl_edge_vs_baseline": pnl_edge,
            "promotion_recommendation": rec,
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
