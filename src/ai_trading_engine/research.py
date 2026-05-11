from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_trading_engine.models import ClosedTrade


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion. Returns (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / float(n)
    z2_n = (z * z) / float(n)
    denom = 1.0 + z2_n
    center = (p + z2_n / 2.0) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2_n / 4.0) / float(n))) / denom
    return float(_clip(center - margin, 0.0, 1.0)), float(_clip(center + margin, 0.0, 1.0))


def _binomial_p_value(wins: int, n: int) -> float:
    """One-tailed p-value for H0: win_rate <= 0.5, normal approx with continuity correction."""
    if n == 0:
        return 1.0
    sigma = math.sqrt(n * 0.25)
    if sigma == 0.0:
        return 1.0
    z = (float(wins) - 0.5 - n * 0.5) / sigma
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _bin_label(confidence: float, bins: int) -> str:
    b = max(2, min(20, int(bins)))
    width = 1.0 / float(b)
    c = _clip(float(confidence), 0.0, 0.999999)
    idx = int(c / width)
    low = idx * width
    high = low + width
    return f"{low:.2f}-{high:.2f}"


@dataclass(frozen=True)
class ResearchSample:
    closed_at: datetime
    direction: str
    regime: str
    confidence: float
    size: int
    pnl: float


def build_trade_dataset(trades: list[ClosedTrade]) -> list[ResearchSample]:
    out: list[ResearchSample] = []
    for trade in trades:
        if trade.closed_at.tzinfo is None:
            closed_at = trade.closed_at.replace(tzinfo=timezone.utc)
        else:
            closed_at = trade.closed_at.astimezone(timezone.utc)
        out.append(
            ResearchSample(
                closed_at=closed_at,
                direction=str(trade.direction),
                regime=str(trade.regime or "unknown"),
                confidence=float(_clip(trade.confidence, 0.0, 1.0)),
                size=max(1, int(trade.size)),
                pnl=float(trade.pnl),
            )
        )
    out.sort(key=lambda x: x.closed_at)
    return out


def _metrics(samples: list[ResearchSample]) -> dict[str, float]:
    if not samples:
        return {
            "count": 0.0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "win_rate_ci95_low": 0.0,
            "win_rate_ci95_high": 0.0,
            "win_rate_p_value": 1.0,
        }

    pnls = [s.pnl for s in samples]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    count = len(samples)
    win_count = len(wins)
    win_rate = win_count / float(count)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    ci_low, ci_high = _wilson_ci(win_count, count)
    p_value = _binomial_p_value(win_count, count)
    return {
        "count": float(count),
        "net_pnl": float(sum(pnls)),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "win_rate_ci95_low": ci_low,
        "win_rate_ci95_high": ci_high,
        "win_rate_p_value": p_value,
    }


def _build_expectancy_maps(samples: list[ResearchSample], bins: int) -> dict[str, dict[str, float]]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def ingest(key: str, pnl: float) -> None:
        sums[key] = float(sums.get(key, 0.0)) + float(pnl)
        counts[key] = int(counts.get(key, 0)) + 1

    for s in samples:
        b = _bin_label(s.confidence, bins)
        ingest(f"rdb:{s.regime}|{s.direction}|{b}", s.pnl)
        ingest(f"db:{s.direction}|{b}", s.pnl)
        ingest(f"b:{b}", s.pnl)
        ingest("all", s.pnl)

    mean: dict[str, float] = {}
    for k, total in sums.items():
        c = max(1, counts.get(k, 0))
        mean[k] = total / float(c)
    countf = {k: float(v) for k, v in counts.items()}
    return {"mean": mean, "count": countf}


def _predict_expectancy(sample: ResearchSample, maps: dict[str, dict[str, float]], bins: int) -> float:
    mean = maps["mean"]
    count = maps["count"]
    b = _bin_label(sample.confidence, bins)

    candidates = [
        (f"rdb:{sample.regime}|{sample.direction}|{b}", 3.0),
        (f"db:{sample.direction}|{b}", 3.0),
        (f"b:{b}", 3.0),
        ("all", 1.0),
    ]
    for key, min_count in candidates:
        if key in mean and count.get(key, 0.0) >= min_count:
            return float(mean[key])
    return 0.0


def _best_threshold(train: list[ResearchSample], maps: dict[str, dict[str, float]], bins: int) -> tuple[float, dict[str, float]]:
    preds = [_predict_expectancy(s, maps, bins) for s in train]
    if not preds:
        return 0.0, _metrics([])
    candidates = sorted(set([0.0] + preds))
    best_thr = 0.0
    best_score = float("-inf")
    best_metrics = _metrics([])
    for thr in candidates:
        selected = [s for s, p in zip(train, preds) if p >= thr]
        m = _metrics(selected)
        # Optimize net pnl first, then expectancy for tie-breaks.
        score = float(m["net_pnl"]) + (0.001 * float(m["expectancy"]))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics


def run_walk_forward(
    samples: list[ResearchSample],
    *,
    folds: int = 4,
    min_train: int = 40,
    min_test: int = 60,
    bins: int = 10,
) -> dict[str, Any]:
    data = sorted(samples, key=lambda x: x.closed_at)
    n = len(data)
    if n < (min_train + min_test):
        return {
            "ok": False,
            "reason": "insufficient_samples",
            "sample_count": n,
            "required_minimum": min_train + min_test,
            "folds": [],
            "aggregate": {},
        }

    use_folds = max(1, min(12, int(folds)))
    usable = n - min_train
    step = max(min_test, usable // use_folds)
    fold_reports: list[dict[str, Any]] = []

    for i in range(use_folds):
        train_end = min_train + (i * step)
        if train_end >= n:
            break
        test_start = train_end
        test_end = min(n, test_start + step)
        if (test_end - test_start) < min_test:
            break

        train = data[:train_end]
        test = data[test_start:test_end]
        maps = _build_expectancy_maps(train, bins=bins)
        threshold, train_best = _best_threshold(train, maps, bins=bins)
        test_preds = [_predict_expectancy(s, maps, bins=bins) for s in test]
        test_selected = [s for s, p in zip(test, test_preds) if p >= threshold]

        fold_reports.append(
            {
                "fold": i + 1,
                "train_count": len(train),
                "test_count": len(test),
                "threshold_expectancy": threshold,
                "train_selected_metrics": train_best,
                "test_selected_metrics": _metrics(test_selected),
                "test_all_metrics": _metrics(test),
                "train_window": {
                    "start": train[0].closed_at.isoformat(),
                    "end": train[-1].closed_at.isoformat(),
                },
                "test_window": {
                    "start": test[0].closed_at.isoformat(),
                    "end": test[-1].closed_at.isoformat(),
                },
            }
        )

    # Aggregate using fold-level metrics weighted by count.
    def weighted_avg(metric: str, key: str) -> float:
        num = 0.0
        den = 0.0
        for f in fold_reports:
            m = f[key]
            w = float(m.get("count", 0.0))
            num += float(m.get(metric, 0.0)) * w
            den += w
        return (num / den) if den > 0 else 0.0

    total_selected = sum(float(f["test_selected_metrics"]["count"]) for f in fold_reports)
    total_test = sum(float(f["test_all_metrics"]["count"]) for f in fold_reports)
    total_selected_pnl = sum(float(f["test_selected_metrics"]["net_pnl"]) for f in fold_reports)
    total_test_pnl = sum(float(f["test_all_metrics"]["net_pnl"]) for f in fold_reports)

    # Pooled win count across folds for aggregate CI.
    total_sel_wins = int(round(sum(
        float(f["test_selected_metrics"]["win_rate"]) * float(f["test_selected_metrics"]["count"])
        for f in fold_reports
    )))
    total_sel_count = int(total_selected)
    sel_ci_low, sel_ci_high = _wilson_ci(total_sel_wins, total_sel_count)
    sel_p_value = _binomial_p_value(total_sel_wins, total_sel_count)

    return {
        "ok": True,
        "sample_count": n,
        "params": {
            "folds": use_folds,
            "min_train": min_train,
            "min_test": min_test,
            "bins": bins,
        },
        "folds": fold_reports,
        "aggregate": {
            "selected_trades_total": total_selected,
            "test_trades_total": total_test,
            "selected_net_pnl_total": total_selected_pnl,
            "test_net_pnl_total": total_test_pnl,
            "selected_expectancy_weighted": weighted_avg("expectancy", "test_selected_metrics"),
            "test_expectancy_weighted": weighted_avg("expectancy", "test_all_metrics"),
            "selected_win_rate_weighted": weighted_avg("win_rate", "test_selected_metrics"),
            "test_win_rate_weighted": weighted_avg("win_rate", "test_all_metrics"),
            "selected_win_rate_ci95_low": sel_ci_low,
            "selected_win_rate_ci95_high": sel_ci_high,
            "selected_win_rate_p_value": sel_p_value,
        },
    }


def save_walk_forward_report(report: dict[str, Any], output_dir: str = "data/research") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"walk_forward_report_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def samples_to_records(samples: list[ResearchSample]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in samples:
        row = asdict(s)
        row["closed_at"] = s.closed_at.isoformat()
        out.append(row)
    return out
