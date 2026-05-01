from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ai_trading_engine.prediction_quality import build_prediction_labels


def _parse_ts(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    if n <= 0:
        return {"count": 0.0, "accuracy": 0.0, "avg_signed_return_bps": 0.0, "brier_score": 0.0}
    wins = 0.0
    signed_total = 0.0
    brier_total = 0.0
    for r in rows:
        signed = float(r.get("signed_return_bps", 0.0))
        conf = max(0.0, min(1.0, float(r.get("confidence", 0.0))))
        y = 1.0 if signed > 0 else 0.0
        wins += y
        signed_total += signed
        brier_total += (conf - y) ** 2
    return {
        "count": float(n),
        "accuracy": wins / float(n),
        "avg_signed_return_bps": signed_total / float(n),
        "brier_score": brier_total / float(n),
    }


def build_model_decay_report(
    samples: list[dict[str, Any]],
    *,
    horizon_minutes: int = 15,
    min_confidence: float = 0.0,
    short_window_days: int = 7,
    long_window_days: int = 30,
    min_labels: int = 40,
    min_accuracy_delta: float = -0.07,
    min_signed_bps_delta: float = -8.0,
    max_brier_delta: float = 0.08,
) -> dict[str, Any]:
    report = build_prediction_labels(
        samples,
        horizons_minutes=(int(horizon_minutes),),
        min_confidence=max(0.0, min(1.0, float(min_confidence))),
    )
    labels = list(report["labels_by_horizon"].get(int(horizon_minutes), []))
    now = datetime.now(tz=timezone.utc)
    short_cut = now - timedelta(days=max(1, int(short_window_days)))
    long_cut = now - timedelta(days=max(2, int(long_window_days)))

    short_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for row in labels:
        ts = _parse_ts(row.get("timestamp"))
        if ts is None:
            continue
        if ts >= short_cut:
            short_rows.append(row)
        if ts >= long_cut:
            long_rows.append(row)

    short_m = _metrics(short_rows)
    long_m = _metrics(long_rows)
    deltas = {
        "accuracy": float(short_m["accuracy"]) - float(long_m["accuracy"]),
        "avg_signed_return_bps": float(short_m["avg_signed_return_bps"]) - float(long_m["avg_signed_return_bps"]),
        "brier_score": float(short_m["brier_score"]) - float(long_m["brier_score"]),
    }
    enough = bool(short_m["count"] >= float(min_labels) and long_m["count"] >= float(min_labels))
    breached_accuracy = enough and deltas["accuracy"] < float(min_accuracy_delta)
    breached_signed = enough and deltas["avg_signed_return_bps"] < float(min_signed_bps_delta)
    breached_brier = enough and deltas["brier_score"] > float(max_brier_delta)
    breached = bool(breached_accuracy or breached_signed or breached_brier)
    reasons: list[str] = []
    if breached_accuracy:
        reasons.append("accuracy_decay")
    if breached_signed:
        reasons.append("signed_return_decay")
    if breached_brier:
        reasons.append("calibration_decay")

    return {
        "ok": True,
        "horizon_minutes": int(horizon_minutes),
        "short_window_days": int(short_window_days),
        "long_window_days": int(long_window_days),
        "min_confidence": float(min_confidence),
        "min_labels": int(min_labels),
        "input_sample_count": int(report["input_sample_count"]),
        "label_count": int(len(labels)),
        "short_window": short_m,
        "long_window": long_m,
        "deltas": deltas,
        "thresholds": {
            "min_accuracy_delta": float(min_accuracy_delta),
            "min_signed_bps_delta": float(min_signed_bps_delta),
            "max_brier_delta": float(max_brier_delta),
        },
        "sufficient_sample": enough,
        "breached": breached,
        "breach_reasons": reasons,
    }
