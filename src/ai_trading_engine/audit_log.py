from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


_LOCK = Lock()


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_prev_hash(state_path: Path) -> str:
    if not state_path.exists():
        return ""
    try:
        return state_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def append_audit_event(log_path: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Append-only JSONL audit with hash chaining.
    Tampering can be detected by replaying hash links.
    """
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state_path = p.with_suffix(p.suffix + ".state")

    with _LOCK:
        prev_hash = _read_prev_hash(state_path)
        event: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
            "prev_hash": prev_hash,
        }
        canonical = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        event_hash = _sha256_hex(canonical)
        record = dict(event)
        record["hash"] = event_hash
        line = json.dumps(record, ensure_ascii=True)
        with p.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        state_path.write_text(event_hash, encoding="utf-8")
        return record
