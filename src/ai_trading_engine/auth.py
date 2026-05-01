from __future__ import annotations

import base64
import hmac


def verify_basic_auth(
    authorization_header: str | None,
    *,
    username: str,
    password: str,
) -> bool:
    if not authorization_header:
        return False
    raw = str(authorization_header).strip()
    if not raw.lower().startswith("basic "):
        return False
    token = raw[6:].strip()
    if not token:
        return False
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except Exception:
        return False
    if ":" not in decoded:
        return False
    got_user, got_pass = decoded.split(":", 1)
    return hmac.compare_digest(got_user, str(username)) and hmac.compare_digest(got_pass, str(password))
