from __future__ import annotations

import statistics
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_trading_engine.llm.parser import parse_decision


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_ts(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _raw_prediction(sample: dict[str, Any]) -> tuple[str, str | None, float, str]:
    raw = str(sample.get("llm_raw") or "")
    if raw:
        try:
            decision = parse_decision(raw)
            if decision.forecast_direction in {"LONG", "SHORT"}:
                return (
                    decision.action,
                    decision.forecast_direction,
                    _clip(float(decision.forecast_confidence), 0.0, 1.0),
                    "forecast",
                )
            return decision.action, decision.direction, _clip(float(decision.confidence), 0.0, 1.0), "trade"
        except Exception:
            pass
    forecast_direction = str(sample.get("forecast_direction")).strip().upper() if sample.get("forecast_direction") else None
    if forecast_direction in {"LONG", "SHORT"}:
        return (
            str(sample.get("action") or "hold").strip().lower(),
            forecast_direction,
            _clip(_float(sample.get("forecast_confidence")), 0.0, 1.0),
            "forecast",
        )
    return (
        str(sample.get("action") or "hold").strip().lower(),
        str(sample.get("direction")).strip().upper() if sample.get("direction") else None,
        _clip(_float(sample.get("confidence")), 0.0, 1.0),
        "trade",
    )


@dataclass(frozen=True)
class _Point:
    timestamp: datetime
    symbol: str
    price: float
    sample: dict[str, Any]


def _points_by_symbol(samples: list[dict[str, Any]]) -> dict[str, list[_Point]]:
    out: dict[str, list[_Point]] = {}
    for sample in samples:
        symbol = str(sample.get("symbol") or "").strip().upper()
        ts = _parse_ts(sample.get("timestamp"))
        price = _float(sample.get("quote_last"))
        if not symbol or ts is None or price <= 0:
            continue
        out.setdefault(symbol, []).append(_Point(timestamp=ts, symbol=symbol, price=price, sample=sample))
    for rows in out.values():
        rows.sort(key=lambda p: p.timestamp)
    return out


def _confidence_bin(confidence: float) -> str:
    c = _clip(confidence, 0.0, 0.999999)
    low = int(c * 10) / 10.0
    high = low + 0.1
    return f"{low:.1f}-{high:.1f}"


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "accuracy": 0.0,
            "avg_signed_return_bps": 0.0,
            "median_signed_return_bps": 0.0,
            "trimmed_mean_signed_return_bps": 0.0,
            "p25_signed_return_bps": 0.0,
            "p75_signed_return_bps": 0.0,
            "tail_loss_bps_p5": 0.0,
            "tail_win_bps_p95": 0.0,
            "avg_forward_return_bps": 0.0,
            "brier_score": 0.0,
            "avg_confidence": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "win_loss_ratio": 0.0,
            "expectancy_bps": 0.0,
        }
    signed = [float(r["signed_return_bps"]) for r in rows]
    forward = [float(r["forward_return_bps"]) for r in rows]
    wins = [1.0 if s > 0 else 0.0 for s in signed]
    conf = [_clip(float(r["confidence"]), 0.0, 1.0) for r in rows]
    brier = sum((p - y) ** 2 for p, y in zip(conf, wins)) / float(len(rows))
    sorted_signed = sorted(signed)
    trim_n = int(len(sorted_signed) * 0.1)
    trimmed = sorted_signed[trim_n:len(sorted_signed) - trim_n] if trim_n > 0 else sorted_signed
    win_returns = [s for s in signed if s > 0.0]
    loss_returns = [s for s in signed if s <= 0.0]
    avg_win = sum(win_returns) / float(len(win_returns)) if win_returns else 0.0
    avg_loss = sum(loss_returns) / float(len(loss_returns)) if loss_returns else 0.0
    return {
        "count": len(rows),
        "accuracy": sum(wins) / float(len(rows)),
        "avg_signed_return_bps": sum(signed) / float(len(rows)),
        "median_signed_return_bps": statistics.median(signed),
        "trimmed_mean_signed_return_bps": sum(trimmed) / float(len(trimmed)) if trimmed else 0.0,
        "p25_signed_return_bps": _percentile(sorted_signed, 0.25),
        "p75_signed_return_bps": _percentile(sorted_signed, 0.75),
        "tail_loss_bps_p5": _percentile(sorted_signed, 0.05),
        "tail_win_bps_p95": _percentile(sorted_signed, 0.95),
        "avg_forward_return_bps": sum(forward) / float(len(rows)),
        "brier_score": brier,
        "avg_confidence": sum(conf) / float(len(rows)),
        "win_count": len(win_returns),
        "loss_count": len(loss_returns),
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "win_loss_ratio": (avg_win / abs(avg_loss)) if avg_loss < 0.0 else 0.0,
        "expectancy_bps": sum(signed) / float(len(rows)),
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return {k: _metric(v) for k, v in sorted(groups.items())}


def _policy_status(metric: dict[str, Any], *, min_count: int, min_accuracy: float, stress_bps: float) -> str:
    count = int(metric.get("count", 0) or 0)
    if count < int(min_count):
        return "probation"
    signed = float(metric.get("avg_signed_return_bps", 0.0) or 0.0)
    accuracy = float(metric.get("accuracy", 0.0) or 0.0)
    if (signed - float(stress_bps)) > 0.0 and accuracy >= float(min_accuracy):
        return "allow"
    return "block"


def _policy_rows(
    rows: list[dict[str, Any]],
    key: str,
    *,
    min_count: int,
    min_accuracy: float,
    stress_bps: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, metric in _group_metrics(rows, key).items():
        signed = float(metric.get("avg_signed_return_bps", 0.0) or 0.0)
        row = dict(metric)
        row["status"] = _policy_status(
            metric,
            min_count=max(1, int(min_count)),
            min_accuracy=max(0.0, min(1.0, float(min_accuracy))),
            stress_bps=max(0.0, float(stress_bps)),
        )
        row["stressed_signed_bps"] = signed - max(0.0, float(stress_bps))
        out[str(label)] = row
    return out


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = _clip(float(q), 0.0, 1.0) * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = pos - lo
    return float((sorted_values[lo] * (1.0 - weight)) + (sorted_values[hi] * weight))


def _calibration_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bin: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_bin.setdefault(str(row.get("confidence_bin") or "unknown"), []).append(row)

    buckets: dict[str, dict[str, Any]] = {}
    weighted_abs_error = 0.0
    total = 0
    for label, items in sorted(by_bin.items()):
        if not items:
            continue
        metric = _metric(items)
        mean_conf = sum(_clip(float(r.get("confidence", 0.0)), 0.0, 1.0) for r in items) / float(len(items))
        realized = float(metric["accuracy"])
        error = realized - mean_conf
        weighted_abs_error += abs(error) * len(items)
        total += len(items)
        buckets[label] = {
            "count": len(items),
            "mean_confidence": mean_conf,
            "realized_accuracy": realized,
            "calibration_error": error,
            "abs_calibration_error": abs(error),
            "avg_signed_return_bps": float(metric["avg_signed_return_bps"]),
            "brier_score": float(metric["brier_score"]),
        }

    return {
        "count": total,
        "weighted_abs_calibration_error": (weighted_abs_error / float(total)) if total else 0.0,
        "by_confidence_bin": buckets,
    }


def _filter_samples_by_quality(samples: list[dict[str, Any]], quality_mode: str) -> list[dict[str, Any]]:
    mode = str(quality_mode or "all").strip().lower()
    if mode == "good_only":
        return [s for s in samples if _sample_is_good(s)]
    return samples


def _sample_is_good(sample: dict[str, Any]) -> bool:
    # Honor explicit quality verdict first.
    if "sample_quality_good" in sample:
        try:
            if not bool(sample.get("sample_quality_good")):
                return False
        except Exception:
            return False
    hard_fail = sample.get("sample_quality_hard_fail_flags")
    if isinstance(hard_fail, dict):
        return not any(bool(v) for v in hard_fail.values())
    flags = sample.get("sample_quality_flags")
    if isinstance(flags, dict):
        return not any(
            bool(flags.get(k))
            for k in ("economic_calendar_error", "finnhub_context_error", "missing_forecast")
        )
    return bool(sample.get("sample_quality_good"))


def build_prediction_labels(
    samples: list[dict[str, Any]],
    *,
    horizons_minutes: tuple[int, ...] = (5, 15, 30),
    min_confidence: float = 0.0,
    quality_mode: str = "all",
    max_quote_age_seconds: float | None = None,
    allowed_session_buckets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    filtered_samples = _filter_samples_by_quality(samples, quality_mode)
    by_symbol = _points_by_symbol(filtered_samples)
    labels_by_horizon: dict[int, list[dict[str, Any]]] = {int(h): [] for h in horizons_minutes}
    eligible = 0
    skipped_no_quote = 0
    skipped_no_prediction = 0

    for symbol, points in by_symbol.items():
        timestamps = [p.timestamp for p in points]
        for point in points:
            sample = point.sample
            if max_quote_age_seconds is not None:
                qa = _float(sample.get("quote_age_seconds"), -1.0)
                if qa >= 0.0 and qa > float(max_quote_age_seconds):
                    continue
            if allowed_session_buckets:
                sb = str(sample.get("session_bucket") or "").strip().lower()
                if sb not in {str(x).strip().lower() for x in allowed_session_buckets}:
                    continue
            action, direction, confidence, prediction_source = _raw_prediction(point.sample)
            if direction not in {"LONG", "SHORT"}:
                skipped_no_prediction += 1
                continue
            if confidence < min_confidence:
                continue
            eligible += 1
            side = 1.0 if direction == "LONG" else -1.0
            for horizon in horizons_minutes:
                target = point.timestamp + timedelta(minutes=int(horizon))
                idx = bisect_left(timestamps, target)
                if idx >= len(points):
                    continue
                future = points[idx]
                forward_bps = ((future.price / point.price) - 1.0) * 10000.0
                signed_bps = forward_bps * side
                feature_patterns = dict(sample.get("feature_patterns") or {})
                feature_structure = dict(sample.get("feature_structure") or {})
                labels_by_horizon[int(horizon)].append(
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "future_timestamp": future.timestamp.isoformat(),
                        "symbol": symbol,
                        "collection_role": sample.get("collection_role"),
                        "llm_provider": sample.get("llm_provider"),
                        "macro_risk_regime": sample.get("macro_risk_regime"),
                        "macro_rate_regime": sample.get("macro_rate_regime"),
                        "macro_provider": sample.get("macro_provider"),
                        "economic_calendar_event_risk": sample.get("economic_calendar_event_risk"),
                        "finnhub_news_risk": sample.get("finnhub_news_risk"),
                        "finnhub_earnings_risk": sample.get("finnhub_earnings_risk"),
                        "finnhub_news_sentiment_label": sample.get("finnhub_news_sentiment_label"),
                        "finnhub_news_sentiment_score": _float(sample.get("finnhub_news_sentiment_score")),
                        "indicator_regime": sample.get("indicator_regime") or "unknown",
                        "feature_volatility_label": sample.get("feature_volatility_label") or "",
                        "feature_volume_label": sample.get("feature_volume_label") or "",
                        "feature_sr_label": sample.get("feature_sr_label") or "",
                        "feature_pattern_bias": sample.get("feature_pattern_bias") or "",
                        "feature_pattern_score": _float(sample.get("feature_pattern_score")),
                        "feature_pattern_summary": sample.get("feature_pattern_summary") or "",
                        "feature_patterns": feature_patterns,
                        "feature_structure_bias": sample.get("feature_structure_bias") or "",
                        "feature_structure_score": _float(sample.get("feature_structure_score")),
                        "feature_structure_summary": sample.get("feature_structure_summary") or "",
                        "feature_structure": feature_structure,
                        "session_segment": feature_structure.get("session_segment") or "unknown",
                        "action": action,
                        "prediction_source": prediction_source,
                        "direction": direction,
                        "confidence": confidence,
                        "confidence_bin": _confidence_bin(confidence),
                        "entry_price": point.price,
                        "future_price": future.price,
                        "forward_return_bps": forward_bps,
                        "signed_return_bps": signed_bps,
                        "correct": signed_bps > 0,
                    }
                )

    total_samples = sum(len(v) for v in by_symbol.values())
    skipped_no_quote = max(0, len(filtered_samples) - total_samples)
    return {
        "quality_mode": str(quality_mode or "all").strip().lower(),
        "input_sample_count": len(samples),
        "filtered_input_sample_count": len(filtered_samples),
        "samples_with_quote": total_samples,
        "eligible_directional_predictions": eligible,
        "eligible_trade_predictions": eligible,
        "skipped_no_quote": skipped_no_quote,
        "skipped_no_prediction": skipped_no_prediction,
        "min_confidence": float(min_confidence),
        "labels_by_horizon": labels_by_horizon,
    }


def build_prediction_quality_report(
    samples: list[dict[str, Any]],
    *,
    horizons_minutes: tuple[int, ...] = (5, 15, 30),
    min_confidence: float = 0.0,
    quality_mode: str = "all",
    max_quote_age_seconds: float | None = None,
    allowed_session_buckets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    label_report = build_prediction_labels(
        samples,
        horizons_minutes=horizons_minutes,
        min_confidence=min_confidence,
        quality_mode=quality_mode,
        max_quote_age_seconds=max_quote_age_seconds,
        allowed_session_buckets=allowed_session_buckets,
    )
    labels_by_horizon = label_report["labels_by_horizon"]
    horizons: dict[str, Any] = {}
    for horizon, labels in labels_by_horizon.items():
        horizons[str(horizon)] = {
            "horizon_minutes": horizon,
            "label_count": len(labels),
            "overall": _metric(labels),
            "calibration": _calibration_diagnostics(labels),
            "by_symbol": _group_metrics(labels, "symbol"),
            "by_direction": _group_metrics(labels, "direction"),
            "by_confidence_bin": _group_metrics(labels, "confidence_bin"),
            "by_macro_risk_regime": _group_metrics(labels, "macro_risk_regime"),
            "by_economic_calendar_event_risk": _group_metrics(labels, "economic_calendar_event_risk"),
            "by_finnhub_news_risk": _group_metrics(labels, "finnhub_news_risk"),
            "by_finnhub_earnings_risk": _group_metrics(labels, "finnhub_earnings_risk"),
            "by_finnhub_news_sentiment_label": _group_metrics(labels, "finnhub_news_sentiment_label"),
            "by_indicator_regime": _group_metrics(labels, "indicator_regime"),
            "by_session_segment": _group_metrics(labels, "session_segment"),
            "by_structure_bias": _group_metrics(labels, "feature_structure_bias"),
        }

    return {
        "ok": True,
        "quality_mode": label_report["quality_mode"],
        "input_sample_count": label_report["input_sample_count"],
        "filtered_input_sample_count": label_report["filtered_input_sample_count"],
        "samples_with_quote": label_report["samples_with_quote"],
        "eligible_directional_predictions": label_report["eligible_directional_predictions"],
        "eligible_trade_predictions": label_report["eligible_trade_predictions"],
        "skipped_no_quote": label_report["skipped_no_quote"],
        "skipped_no_prediction": label_report["skipped_no_prediction"],
        "min_confidence": label_report["min_confidence"],
        "horizons": horizons,
    }


def _ablation_metric(rows: list[dict[str, Any]], enabled_fn, *, min_count: int) -> dict[str, Any]:
    enabled = [r for r in rows if enabled_fn(r)]
    baseline = [r for r in rows if not enabled_fn(r)]
    enabled_m = _metric(enabled)
    baseline_m = _metric(baseline)
    return {
        "enabled": enabled_m,
        "baseline": baseline_m,
        "delta_accuracy": float(enabled_m["accuracy"]) - float(baseline_m["accuracy"]),
        "delta_signed_return_bps": float(enabled_m["avg_signed_return_bps"]) - float(baseline_m["avg_signed_return_bps"]),
        "delta_brier_score": float(enabled_m["brier_score"]) - float(baseline_m["brier_score"]),
        "sufficient_sample": bool(enabled_m["count"] >= min_count and baseline_m["count"] >= min_count),
    }


def build_feature_ablation_report(
    samples: list[dict[str, Any]],
    *,
    horizon_minutes: int = 15,
    min_confidence: float = 0.0,
    min_count: int = 20,
    quality_mode: str = "all",
    max_quote_age_seconds: float | None = None,
    allowed_session_buckets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    label_report = build_prediction_labels(
        samples,
        horizons_minutes=(int(horizon_minutes),),
        min_confidence=min_confidence,
        quality_mode=quality_mode,
        max_quote_age_seconds=max_quote_age_seconds,
        allowed_session_buckets=allowed_session_buckets,
    )
    rows = label_report["labels_by_horizon"].get(int(horizon_minutes), [])
    features = {
        "macro_fred": lambda r: str(r.get("macro_provider") or "") == "fred",
        "macro_non_unknown_risk": lambda r: str(r.get("macro_risk_regime") or "") not in {"", "unknown"},
        "economic_calendar_active": lambda r: str(r.get("economic_calendar_event_risk") or "") not in {"", "clear", "disabled"},
        "high_impact_upcoming": lambda r: "high_impact" in str(r.get("economic_calendar_event_risk") or ""),
        "news_active": lambda r: str(r.get("finnhub_news_risk") or "") not in {"", "clear", "disabled"},
        "earnings_active": lambda r: str(r.get("finnhub_earnings_risk") or "") not in {"", "clear", "disabled"},
        "sentiment_non_neutral": lambda r: str(r.get("finnhub_news_sentiment_label") or "neutral") != "neutral",
        "sentiment_bullish": lambda r: str(r.get("finnhub_news_sentiment_label") or "") == "bullish",
        "sentiment_bearish": lambda r: str(r.get("finnhub_news_sentiment_label") or "") == "bearish",
    }
    ablations = {
        name: _ablation_metric(rows, fn, min_count=max(1, int(min_count)))
        for name, fn in features.items()
    }
    ranked = sorted(
        (
            {
                "feature": name,
                "delta_signed_return_bps": report["delta_signed_return_bps"],
                "delta_accuracy": report["delta_accuracy"],
                "delta_brier_score": report["delta_brier_score"],
                "enabled_count": report["enabled"]["count"],
                "baseline_count": report["baseline"]["count"],
                "sufficient_sample": report["sufficient_sample"],
            }
            for name, report in ablations.items()
        ),
        key=lambda r: float(r["delta_signed_return_bps"]),
        reverse=True,
    )
    return {
        "ok": True,
        "quality_mode": label_report["quality_mode"],
        "horizon_minutes": int(horizon_minutes),
        "min_confidence": float(min_confidence),
        "min_count": int(min_count),
        "input_sample_count": label_report["input_sample_count"],
        "filtered_input_sample_count": label_report["filtered_input_sample_count"],
        "label_count": len(rows),
        "overall": _metric(rows),
        "ablations": ablations,
        "ranked": ranked,
        "note": "Positive deltas indicate labels performed better when the feature condition was present. Treat insufficient samples as directional only.",
    }


def build_confidence_control_report(
    samples: list[dict[str, Any]],
    *,
    horizon_minutes: int = 15,
    quality_mode: str = "good_only",
    min_bin_count: int = 20,
    min_symbol_labels: int = 40,
    min_threshold_labels: int = 30,
    default_min_confidence: float = 0.55,
    policy_min_accuracy: float = 0.50,
    policy_stress_bps: float = 0.0,
    max_quote_age_seconds: float | None = None,
    allowed_session_buckets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    label_report = build_prediction_labels(
        samples,
        horizons_minutes=(max(1, int(horizon_minutes)),),
        min_confidence=0.0,
        quality_mode=quality_mode,
        max_quote_age_seconds=max_quote_age_seconds,
        allowed_session_buckets=allowed_session_buckets,
    )
    rows = label_report["labels_by_horizon"].get(max(1, int(horizon_minutes)), [])
    if not rows:
        return {
            "ok": True,
            "quality_mode": str(quality_mode or "good_only").strip().lower(),
            "horizon_minutes": int(horizon_minutes),
            "overall_label_count": 0,
            "default_min_confidence": float(default_min_confidence),
            "global_threshold": float(default_min_confidence),
            "symbol_controls": {},
            "policy": {
                "min_accuracy": float(policy_min_accuracy),
                "stress_bps": float(policy_stress_bps),
                "by_symbol": {},
            },
        }

    def _choose_threshold(items: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
        candidates = sorted({_clip(float(r.get("confidence", 0.0)), 0.0, 1.0) for r in items})
        best_thr = float(default_min_confidence)
        best_summary = {
            "count": 0,
            "accuracy": 0.0,
            "avg_signed_return_bps": 0.0,
            "brier_score": 0.0,
        }
        for thr in candidates:
            selected = [r for r in items if float(r.get("confidence", 0.0)) >= thr]
            if len(selected) < int(min_threshold_labels):
                continue
            m = _metric(selected)
            if (
                float(m["avg_signed_return_bps"]) > float(best_summary["avg_signed_return_bps"])
                or (
                    float(m["avg_signed_return_bps"]) == float(best_summary["avg_signed_return_bps"])
                    and int(m["count"]) > int(best_summary["count"])
                )
            ):
                best_thr = float(thr)
                best_summary = m
        return best_thr, best_summary

    global_threshold, global_summary = _choose_threshold(rows)
    symbols = sorted({str(r.get("symbol") or "").upper() for r in rows if r.get("symbol")})
    symbol_controls: dict[str, Any] = {}
    symbol_policy: dict[str, Any] = {}
    for sym in symbols:
        sym_rows = [r for r in rows if str(r.get("symbol") or "").upper() == sym]
        calibration: dict[str, Any] = {}
        for bin_label, bin_rows in sorted(_group_metrics(sym_rows, "confidence_bin").items()):
            if int(bin_rows.get("count", 0) or 0) < int(min_bin_count):
                continue
            calibration[bin_label] = {
                "count": int(bin_rows["count"]),
                "raw_confidence_mid": float(bin_label.split("-")[0]) + 0.05,
                "calibrated_confidence": _clip(float(bin_rows["accuracy"]), 0.0, 1.0),
                "avg_signed_return_bps": float(bin_rows["avg_signed_return_bps"]),
                "brier_score": float(bin_rows["brier_score"]),
            }
        if len(sym_rows) < int(min_symbol_labels):
            symbol_controls[sym] = {
                "label_count": len(sym_rows),
                "min_confidence": float(global_threshold),
                "calibration": calibration,
                "source": "global_fallback_low_sample",
                "selected_metrics": global_summary,
            }
            symbol_policy[sym] = {
                "label_count": len(sym_rows),
                "source": "global_fallback_low_sample",
                "by_confidence_bin": _policy_rows(
                    sym_rows,
                    "confidence_bin",
                    min_count=max(1, int(min_bin_count)),
                    min_accuracy=float(policy_min_accuracy),
                    stress_bps=float(policy_stress_bps),
                ),
                "by_direction": _policy_rows(
                    sym_rows,
                    "direction",
                    min_count=max(1, int(min_symbol_labels)),
                    min_accuracy=float(policy_min_accuracy),
                    stress_bps=float(policy_stress_bps),
                ),
            }
            continue
        thr, summary = _choose_threshold(sym_rows)
        symbol_controls[sym] = {
            "label_count": len(sym_rows),
            "min_confidence": float(thr),
            "calibration": calibration,
            "source": "symbol",
            "selected_metrics": summary,
        }
        symbol_policy[sym] = {
            "label_count": len(sym_rows),
            "source": "symbol",
            "by_confidence_bin": _policy_rows(
                sym_rows,
                "confidence_bin",
                min_count=max(1, int(min_bin_count)),
                min_accuracy=float(policy_min_accuracy),
                stress_bps=float(policy_stress_bps),
            ),
            "by_direction": _policy_rows(
                sym_rows,
                "direction",
                min_count=max(1, int(min_symbol_labels)),
                min_accuracy=float(policy_min_accuracy),
                stress_bps=float(policy_stress_bps),
            ),
        }

    return {
        "ok": True,
        "quality_mode": str(quality_mode or "good_only").strip().lower(),
        "horizon_minutes": int(horizon_minutes),
        "overall_label_count": len(rows),
        "default_min_confidence": float(default_min_confidence),
        "global_threshold": float(global_threshold),
        "global_threshold_metrics": global_summary,
        "symbol_controls": symbol_controls,
        "policy": {
            "min_accuracy": max(0.0, min(1.0, float(policy_min_accuracy))),
            "stress_bps": max(0.0, float(policy_stress_bps)),
            "by_symbol": symbol_policy,
        },
    }
