from __future__ import annotations

import json
import re
from dataclasses import asdict

from ai_trading_engine.models import AiDecision


FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("No JSON object found in LLM output")

    # Common case: strict JSON payload.
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    # Remove markdown fences if present.
    unwrapped = FENCE_RE.sub("", raw).strip()
    if unwrapped and unwrapped != raw:
        try:
            payload = json.loads(unwrapped)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    # Scan for balanced JSON objects and parse the first valid dict.
    for obj in _iter_brace_objects(unwrapped or raw):
        try:
            payload = json.loads(obj)
            if isinstance(payload, dict):
                return payload
        except Exception:
            repaired = _repair_json_like(obj)
            if repaired is not None:
                return repaired

    kv_payload = _parse_key_value_block(unwrapped or raw)
    if kv_payload is not None:
        return kv_payload

    raise ValueError("No JSON object found in LLM output")


def _iter_brace_objects(text: str):
    start = -1
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def _repair_json_like(text: str) -> dict | None:
    s = text.strip()
    if not s:
        return None
    # Convert python-style dicts / JSON-ish output into valid JSON.
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
    # Convert single quoted keys/values to double quotes.
    s = re.sub(r"(?<!\\)'", '"', s)
    try:
        payload = json.loads(s)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _parse_key_value_block(text: str) -> dict | None:
    # Handles lines like:
    # action: hold
    # confidence: 0.55
    out: dict[str, object] = {}
    seen = False
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip('"').strip("'").lower()
        value = value.strip().strip('"').strip("'")
        if key in {"action", "direction", "reasoning", "forecast_direction", "confidence_source", "forecast_confidence_source"}:
            out[key] = value
            seen = True
        elif key in {"confidence", "forecast_confidence"}:
            out[key] = _to_float(value, 0.0)
            seen = True
        elif key in {"size", "sl_ticks", "tp_ticks", "forecast_horizon_minutes"}:
            out[key] = _to_int(value, 0)
            seen = True
    return out if seen else None


def parse_decision(text: str) -> AiDecision:
    payload = extract_json(text)

    action = str(payload.get("action", "hold")).strip().lower()
    raw_direction = payload.get("direction")
    direction = str(raw_direction).strip().upper() if raw_direction is not None else None
    confidence = _to_float(payload.get("confidence", 0.0), 0.0)
    size = _to_int(payload.get("size", 1), 1)
    sl_ticks = _to_int(payload.get("sl_ticks", 0), 0)
    tp_ticks = _to_int(payload.get("tp_ticks", 0), 0)
    reasoning = str(payload.get("reasoning", ""))
    confidence_source = str(payload.get("confidence_source", "model") or "model").strip().lower()
    raw_forecast_direction = payload.get("forecast_direction")
    forecast_direction = (
        str(raw_forecast_direction).strip().upper()
        if raw_forecast_direction is not None
        else direction
    )
    forecast_confidence = _to_float(payload.get("forecast_confidence", confidence), confidence)
    forecast_horizon_minutes = _to_int(payload.get("forecast_horizon_minutes", 15), 15)
    forecast_confidence_source = str(
        payload.get("forecast_confidence_source", confidence_source) or confidence_source or "model"
    ).strip().lower()

    if action not in {"trade", "hold"}:
        action = "hold"
    if direction not in {"LONG", "SHORT"}:
        direction = None
    if forecast_direction not in {"LONG", "SHORT"}:
        forecast_direction = None
    forecast_confidence = max(0.0, min(1.0, forecast_confidence))
    forecast_horizon_minutes = max(1, min(390, forecast_horizon_minutes))

    return AiDecision(
        action=action,
        direction=direction,
        confidence=confidence,
        size=size,
        sl_ticks=sl_ticks,
        tp_ticks=tp_ticks,
        reasoning=reasoning,
        forecast_direction=forecast_direction,
        forecast_confidence=forecast_confidence,
        forecast_horizon_minutes=forecast_horizon_minutes,
        confidence_source=confidence_source or "model",
        forecast_confidence_source=forecast_confidence_source or confidence_source or "model",
    )


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def decision_to_json(decision: AiDecision) -> str:
    return json.dumps(asdict(decision), ensure_ascii=True)
