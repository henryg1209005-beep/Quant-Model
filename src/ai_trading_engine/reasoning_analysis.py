from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "them",
    "their", "our", "my", "your", "his", "her", "as", "if", "so", "than",
    "then", "when", "where", "which", "who", "what", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "s", "t",
    "just", "into", "also", "very", "too", "same",
})


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) >= 2]


def _extract_phrases(text: str) -> set[str]:
    tokens = _tokenize(text)
    phrases: set[str] = set()
    for tok in tokens:
        if len(tok) >= 3 and tok not in _STOPWORDS:
            phrases.add(tok)
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if not (a in _STOPWORDS and b in _STOPWORDS):
            phrases.add(f"{a} {b}")
    return phrases


def _chi2_p(a: int, b: int, c: int, d: int) -> float:
    """2×2 chi-squared with Yates correction, 1 df. Two-tailed p-value."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        return 1.0
    ad_bc = abs(float(a) * float(d) - float(b) * float(c))
    corrected = max(0.0, ad_bc - float(n) / 2.0)
    num = float(n) * corrected ** 2
    denom = float(r1) * float(r2) * float(c1) * float(c2)
    if denom == 0.0:
        return 1.0
    chi2 = num / denom
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


def analyze_reasoning(
    trades: list[dict[str, Any]],
    *,
    min_count: int = 5,
    top_n: int = 40,
) -> dict[str, Any]:
    """
    Mine n-gram phrases from LLM reasoning strings and correlate each phrase
    with trade outcome (win/loss). Returns a ranked leaderboard with chi-squared
    significance so results are auditable and reproducible from raw trade data.
    """
    valid = [
        t for t in trades
        if str(t.get("reasoning", "")).strip()
        and t.get("pnl") is not None
    ]
    total = len(trades)
    analyzed = len(valid)

    if analyzed < max(1, int(min_count)):
        return {
            "ok": False,
            "reason": "insufficient_trades_with_reasoning",
            "analyzed_count": analyzed,
            "total_count": total,
            "overall_win_rate": 0.0,
            "phrase_count": 0,
            "phrases": [],
            "generated_at": _utc_now_iso(),
        }

    wins_total = sum(1 for t in valid if float(t.get("pnl", 0)) > 0)
    overall_win_rate = wins_total / float(analyzed)

    phrase_wins: dict[str, int] = {}
    phrase_count: dict[str, int] = {}

    for t in valid:
        phrases = _extract_phrases(str(t.get("reasoning", "")))
        is_win = float(t.get("pnl", 0)) > 0
        for phrase in phrases:
            phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
            if is_win:
                phrase_wins[phrase] = phrase_wins.get(phrase, 0) + 1

    rows: list[dict[str, Any]] = []
    for phrase, cnt in phrase_count.items():
        if cnt < int(min_count):
            continue
        win_c = phrase_wins.get(phrase, 0)
        loss_c = cnt - win_c
        no_cnt = analyzed - cnt
        no_win = wins_total - win_c
        no_loss = no_cnt - no_win

        win_rate = win_c / float(cnt)
        lift = (win_rate - overall_win_rate) / max(overall_win_rate, 1e-6)
        p_value = _chi2_p(win_c, loss_c, no_win, no_loss)

        rows.append({
            "phrase": phrase,
            "count": cnt,
            "win_count": win_c,
            "loss_count": loss_c,
            "win_rate": round(win_rate, 4),
            "lift": round(lift, 4),
            "p_value": round(p_value, 4),
        })

    rows.sort(key=lambda x: abs(float(x["lift"])), reverse=True)

    return {
        "ok": True,
        "analyzed_count": analyzed,
        "total_count": total,
        "overall_win_rate": round(overall_win_rate, 4),
        "phrase_count": len(rows),
        "phrases": rows[: int(top_n) * 2],
        "generated_at": _utc_now_iso(),
    }


def save_reasoning_report(report: dict[str, Any], output_dir: str = "data/research") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"reasoning_analysis_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
