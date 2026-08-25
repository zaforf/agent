"""Small in-process latency recorder for the personal assistant UI/debugging."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any

_MAX_SAMPLES = 200
_samples: deque[dict[str, Any]] = deque(maxlen=_MAX_SAMPLES)
_lock = Lock()


def record(sample: dict[str, Any]) -> None:
    """Store one completed turn, bounded so telemetry cannot become a datastore."""
    item = dict(sample)
    item.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    with _lock:
        _samples.append(item)


def recent(limit: int = 50) -> list[dict[str, Any]]:
    """Return newest samples first, capped to the in-process retention window."""
    limit = max(1, min(int(limit), _MAX_SAMPLES))
    with _lock:
        return list(reversed(list(_samples)[-limit:]))


def clear() -> None:
    """Clear samples; intended for tests and local debugging."""
    with _lock:
        _samples.clear()
