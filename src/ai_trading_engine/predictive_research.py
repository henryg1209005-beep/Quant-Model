from __future__ import annotations

import json
import math
from bisect import bisect_left
from dataclasses import dataclass
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


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    q = _clip(p, 1e-6, 1.0 - 1e-6)
    return math.log(q / (1.0 - q))


def _quantile_thresholds(values: list[float], max_bins: int) -> list[float]:
    if not values:
        return []
    uniq = sorted(set(values))
    if len(uniq) <= 2:
        return uniq[:-1]
    bins = max(2, min(32, int(max_bins)))
    out: list[float] = []
    n = len(uniq)
    for i in range(1, bins):
        idx = int((i / bins) * (n - 1))
        out.append(uniq[idx])
    return sorted(set(out))


@dataclass(frozen=True)
class PredictiveSample:
    closed_at: datetime
    pnl: float
    confidence: float
    size: int
    direction: str
    regime: str

    @property
    def y(self) -> int:
        return 1 if self.pnl > 0 else 0


@dataclass(frozen=True)
class _Stump:
    feature_idx: int
    threshold: float
    left_value: float
    right_value: float


class GradientBoostedStumpsClassifier:
    def __init__(self, n_estimators: int = 80, learning_rate: float = 0.1, max_bins: int = 16) -> None:
        self.n_estimators = max(1, int(n_estimators))
        self.learning_rate = _clip(float(learning_rate), 0.01, 1.0)
        self.max_bins = max(2, min(32, int(max_bins)))
        self.base_logit = 0.0
        self.stumps: list[_Stump] = []

    def _predict_margin_one(self, x: list[float]) -> float:
        m = self.base_logit
        for s in self.stumps:
            m += self.learning_rate * (s.left_value if x[s.feature_idx] <= s.threshold else s.right_value)
        return m

    def predict_proba(self, x_rows: list[list[float]]) -> list[float]:
        return [_sigmoid(self._predict_margin_one(x)) for x in x_rows]

    def fit(self, x_rows: list[list[float]], y: list[int]) -> None:
        n = len(x_rows)
        if n == 0:
            self.base_logit = 0.0
            self.stumps = []
            return
        p0 = sum(y) / float(n)
        self.base_logit = _logit(p0 if 0.0 < p0 < 1.0 else 0.5)
        self.stumps = []

        margins = [self.base_logit for _ in range(n)]
        m_features = len(x_rows[0]) if x_rows and x_rows[0] else 0
        if m_features == 0:
            return

        for _ in range(self.n_estimators):
            probs = [_sigmoid(m) for m in margins]
            residuals = [float(y[i]) - probs[i] for i in range(n)]

            best: _Stump | None = None
            best_score = float("-inf")
            for j in range(m_features):
                vals = [row[j] for row in x_rows]
                thresholds = _quantile_thresholds(vals, self.max_bins)
                for thr in thresholds:
                    left_idx = [i for i, v in enumerate(vals) if v <= thr]
                    right_idx = [i for i, v in enumerate(vals) if v > thr]
                    if not left_idx or not right_idx:
                        continue

                    left_mean = sum(residuals[i] for i in left_idx) / float(len(left_idx))
                    right_mean = sum(residuals[i] for i in right_idx) / float(len(right_idx))
                    score = (
                        sum((residuals[i] - left_mean) ** 2 for i in left_idx)
                        + sum((residuals[i] - right_mean) ** 2 for i in right_idx)
                    )
                    score = -score
                    if score > best_score:
                        best_score = score
                        best = _Stump(
                            feature_idx=j,
                            threshold=float(thr),
                            left_value=float(left_mean),
                            right_value=float(right_mean),
                        )
            if best is None:
                break
            self.stumps.append(best)
            for i, row in enumerate(x_rows):
                margins[i] += self.learning_rate * (best.left_value if row[best.feature_idx] <= best.threshold else best.right_value)


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.x: list[float] = []
        self.y: list[float] = []
        self._fitted = False

    def fit(self, probs: list[float], labels: list[int]) -> None:
        if not probs:
            self.x = [0.0, 1.0]
            self.y = [0.5, 0.5]
            self._fitted = True
            return
        pairs = sorted(zip([_clip(float(p), 0.0, 1.0) for p in probs], labels), key=lambda t: t[0])
        xs = [p for p, _ in pairs]
        ys = [float(y) for _, y in pairs]

        # Pool Adjacent Violators algorithm.
        blocks = [{"sum": ys[i], "count": 1.0, "start": i, "end": i} for i in range(len(ys))]
        i = 0
        while i < len(blocks) - 1:
            avg_i = blocks[i]["sum"] / blocks[i]["count"]
            avg_n = blocks[i + 1]["sum"] / blocks[i + 1]["count"]
            if avg_i <= avg_n:
                i += 1
                continue
            blocks[i]["sum"] += blocks[i + 1]["sum"]
            blocks[i]["count"] += blocks[i + 1]["count"]
            blocks[i]["end"] = blocks[i + 1]["end"]
            del blocks[i + 1]
            if i > 0:
                i -= 1

        cal_x: list[float] = []
        cal_y: list[float] = []
        for b in blocks:
            start = int(b["start"])
            end = int(b["end"])
            px = xs[(start + end) // 2]
            py = b["sum"] / b["count"]
            cal_x.append(px)
            cal_y.append(_clip(py, 0.0, 1.0))

        self.x = cal_x
        self.y = cal_y
        self._fitted = True

    def predict(self, probs: list[float]) -> list[float]:
        if not self._fitted or not self.x:
            return [_clip(float(p), 0.0, 1.0) for p in probs]
        out: list[float] = []
        for p in probs:
            q = _clip(float(p), 0.0, 1.0)
            idx = bisect_left(self.x, q)
            if idx <= 0:
                out.append(self.y[0])
            elif idx >= len(self.x):
                out.append(self.y[-1])
            else:
                x0, x1 = self.x[idx - 1], self.x[idx]
                y0, y1 = self.y[idx - 1], self.y[idx]
                if x1 <= x0:
                    out.append(y1)
                else:
                    t = (q - x0) / (x1 - x0)
                    out.append(_clip(y0 + t * (y1 - y0), 0.0, 1.0))
        return out


def build_predictive_dataset(trades: list[ClosedTrade]) -> list[PredictiveSample]:
    out: list[PredictiveSample] = []
    for t in trades:
        closed_at = t.closed_at if t.closed_at.tzinfo is not None else t.closed_at.replace(tzinfo=timezone.utc)
        out.append(
            PredictiveSample(
                closed_at=closed_at.astimezone(timezone.utc),
                pnl=float(t.pnl),
                confidence=float(_clip(t.confidence, 0.0, 1.0)),
                size=max(1, int(t.size)),
                direction=str(t.direction or "LONG"),
                regime=str(t.regime or "unknown"),
            )
        )
    out.sort(key=lambda s: s.closed_at)
    return out


def _feature_matrix(samples: list[PredictiveSample], regime_levels: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    regime_index = {r: i for i, r in enumerate(regime_levels)}
    for s in samples:
        row = [
            float(s.confidence),
            float(s.size),
            1.0 if s.direction == "LONG" else 0.0,
            1.0 if s.direction == "SHORT" else 0.0,
        ]
        reg = [0.0] * len(regime_levels)
        idx = regime_index.get(s.regime)
        if idx is not None:
            reg[idx] = 1.0
        row.extend(reg)
        out.append(row)
    return out


def _fold_metrics(samples: list[PredictiveSample], probs: list[float], threshold: float) -> dict[str, float]:
    selected = [s for s, p in zip(samples, probs) if p >= threshold]
    pnls = [s.pnl for s in selected]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    count = len(selected)
    win_count = len(wins)
    win_rate = (win_count / float(count)) if count else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss) if count else 0.0
    ci_low, ci_high = _wilson_ci(win_count, count)
    p_value = _binomial_p_value(win_count, count)
    return {
        "count": float(count),
        "net_pnl": float(sum(pnls)),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "win_rate_ci95_low": ci_low,
        "win_rate_ci95_high": ci_high,
        "win_rate_p_value": p_value,
    }


def _best_threshold(samples: list[PredictiveSample], probs: list[float]) -> tuple[float, dict[str, float]]:
    candidates = sorted(set([0.0] + [_clip(float(p), 0.0, 1.0) for p in probs]))
    best_thr = 0.5
    best_metrics = _fold_metrics(samples, probs, threshold=best_thr)
    best_score = best_metrics["net_pnl"] + (0.001 * best_metrics["expectancy"])
    for thr in candidates:
        m = _fold_metrics(samples, probs, threshold=thr)
        score = m["net_pnl"] + (0.001 * m["expectancy"])
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics


def run_predictive_walk_forward(
    samples: list[PredictiveSample],
    *,
    folds: int = 4,
    min_train: int = 100,
    min_test: int = 60,
    n_estimators: int = 80,
    learning_rate: float = 0.1,
    max_bins: int = 16,
) -> dict[str, Any]:
    data = sorted(samples, key=lambda x: x.closed_at)
    n = len(data)
    required = max(1, int(min_train)) + max(1, int(min_test))
    if n < required:
        return {
            "ok": False,
            "reason": "insufficient_samples",
            "sample_count": n,
            "required_minimum": required,
            "folds": [],
            "aggregate": {},
        }

    use_folds = max(1, min(12, int(folds)))
    usable = n - max(1, int(min_train))
    step = max(max(1, int(min_test)), usable // use_folds)
    fold_reports: list[dict[str, Any]] = []

    for i in range(use_folds):
        train_end = int(min_train) + (i * step)
        if train_end >= n:
            break
        test_start = train_end
        test_end = min(n, test_start + step)
        if (test_end - test_start) < int(min_test):
            break

        train = data[:train_end]
        test = data[test_start:test_end]
        split_idx = max(5, int(len(train) * 0.8))
        if split_idx >= len(train):
            split_idx = len(train) - 1
        fit_train = train[:split_idx]
        calib_train = train[split_idx:]

        regime_levels = sorted(set(s.regime for s in fit_train))
        x_fit = _feature_matrix(fit_train, regime_levels)
        y_fit = [s.y for s in fit_train]
        model = GradientBoostedStumpsClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_bins=max_bins,
        )
        model.fit(x_fit, y_fit)

        x_cal = _feature_matrix(calib_train, regime_levels)
        raw_cal = model.predict_proba(x_cal)
        y_cal = [s.y for s in calib_train]
        calibrator = IsotonicCalibrator()
        calibrator.fit(raw_cal, y_cal)

        # Use calib_train for threshold selection: GBM was not fit on it, so it is genuinely held-out.
        calib_model_probs = calibrator.predict(raw_cal)
        model_thr, model_train_metrics = _best_threshold(calib_train, calib_model_probs)

        baseline_probs_calib = [_clip(s.confidence, 0.0, 1.0) for s in calib_train]
        base_thr, base_train_metrics = _best_threshold(calib_train, baseline_probs_calib)

        x_test = _feature_matrix(test, regime_levels)
        test_model_probs = calibrator.predict(model.predict_proba(x_test))
        test_baseline_probs = [_clip(s.confidence, 0.0, 1.0) for s in test]

        fold_reports.append(
            {
                "fold": i + 1,
                "train_count": len(train),
                "test_count": len(test),
                "model_threshold": model_thr,
                "baseline_threshold": base_thr,
                "model_train_selected_metrics": model_train_metrics,
                "baseline_train_selected_metrics": base_train_metrics,
                "model_test_selected_metrics": _fold_metrics(test, test_model_probs, model_thr),
                "baseline_test_selected_metrics": _fold_metrics(test, test_baseline_probs, base_thr),
                "test_all_metrics": _fold_metrics(test, [1.0] * len(test), 0.0),
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

    def _sum_metric(key: str, metric: str) -> float:
        return float(sum(float(f[key][metric]) for f in fold_reports))

    def _weighted_avg(key: str, metric: str) -> float:
        num = 0.0
        den = 0.0
        for f in fold_reports:
            w = float(f[key]["count"])
            num += float(f[key][metric]) * w
            den += w
        return (num / den) if den > 0 else 0.0

    model_net = _sum_metric("model_test_selected_metrics", "net_pnl")
    base_net = _sum_metric("baseline_test_selected_metrics", "net_pnl")
    model_count = _sum_metric("model_test_selected_metrics", "count")
    base_count = _sum_metric("baseline_test_selected_metrics", "count")

    # Pooled CI across all model test folds.
    model_total_wins = int(round(sum(
        float(f["model_test_selected_metrics"]["win_rate"]) * float(f["model_test_selected_metrics"]["count"])
        for f in fold_reports
    )))
    model_total_count = int(model_count)
    model_win_ci_low, model_win_ci_high = _wilson_ci(model_total_wins, model_total_count)
    model_win_p_value = _binomial_p_value(model_total_wins, model_total_count)

    promote = bool(
        len(fold_reports) > 0
        and model_net > base_net
        and _weighted_avg("model_test_selected_metrics", "expectancy")
        >= _weighted_avg("baseline_test_selected_metrics", "expectancy")
        and model_count >= max(1.0, 0.5 * base_count)
        and model_win_ci_low > 0.50
    )

    return {
        "ok": True,
        "sample_count": n,
        "params": {
            "folds": use_folds,
            "min_train": int(min_train),
            "min_test": int(min_test),
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "max_bins": int(max_bins),
        },
        "folds": fold_reports,
        "aggregate": {
            "model_selected_trades_total": model_count,
            "baseline_selected_trades_total": base_count,
            "model_selected_net_pnl_total": model_net,
            "baseline_selected_net_pnl_total": base_net,
            "model_selected_expectancy_weighted": _weighted_avg("model_test_selected_metrics", "expectancy"),
            "baseline_selected_expectancy_weighted": _weighted_avg("baseline_test_selected_metrics", "expectancy"),
            "model_selected_win_rate_weighted": _weighted_avg("model_test_selected_metrics", "win_rate"),
            "baseline_selected_win_rate_weighted": _weighted_avg("baseline_test_selected_metrics", "win_rate"),
            "model_selected_win_rate_ci95_low": model_win_ci_low,
            "model_selected_win_rate_ci95_high": model_win_ci_high,
            "model_selected_win_rate_p_value": model_win_p_value,
            "promotion_recommendation": "promote" if promote else "hold_current",
        },
    }


def save_predictive_report(report: dict[str, Any], output_dir: str = "data/research") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"predictive_walk_forward_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path
