from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_notification(path: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": str(event_type),
        "payload": payload,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    return row


def list_notifications(path: str, limit: int = 50) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines()[-max(1, int(limit)) :]:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        except Exception:
            continue
    return list(reversed(rows))
