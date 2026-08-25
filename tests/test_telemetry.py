from __future__ import annotations

import telemetry


def test_recent_returns_newest_first_and_is_bounded():
    telemetry.clear()
    for i in range(3):
        telemetry.record({"n": i})

    assert [row["n"] for row in telemetry.recent(2)] == [2, 1]
    assert "recorded_at" in telemetry.recent(1)[0]
    telemetry.clear()


def test_recent_limit_is_clamped():
    telemetry.clear()
    telemetry.record({"n": 1})
    assert telemetry.recent(0)
    telemetry.clear()
