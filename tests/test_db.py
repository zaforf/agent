"""SQLite roundtrip + display-history collapse tests.

All tests use the `tmp_db` fixture so nothing touches data/history.db.
"""
from __future__ import annotations

import pytest

import db


def _sample_turn(user="hi"):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc_1", "type": "function",
             "function": {"name": "recall", "arguments": '{"query":"x"}'}},
        ]},
        {"role": "tool", "name": "recall", "tool_call_id": "tc_1", "content": "result A"},
        {"role": "assistant", "content": "Final answer."},
    ]


def test_append_turn_round_trip(tmp_db):
    sid = "s1"
    turn = _sample_turn()
    db.append_turn(sid, "hi", turn)
    got = db.get_history(sid)
    assert got == turn, f"round-trip mismatch:\n  want={turn}\n  got={got}"


def test_multiple_turns_concatenate(tmp_db):
    sid = "s2"
    db.append_turn(sid, "one", _sample_turn("one"))
    db.append_turn(sid, "two", _sample_turn("two"))
    got = db.get_history(sid)
    # 4 messages per turn × 2 turns = 8
    assert len(got) == 8
    assert got[0]["content"] == "one"
    assert got[4]["content"] == "two"


def test_display_history_collapses_tool_turn(tmp_db):
    sid = "s3"
    db.append_turn(sid, "hi", _sample_turn("hi"))
    disp = db.get_display_history(sid)
    assert len(disp) == 2
    assert disp[0] == {"role": "user", "content": "hi"}
    assert disp[1]["role"] == "assistant"
    assert disp[1]["content"] == "Final answer."
    step_types = [s["type"] for s in disp[1]["steps"]]
    assert step_types == ["tool_call", "tool_result"]
    assert disp[1]["steps"][0]["name"] == "recall"
    assert disp[1]["steps"][0]["args"] == {"query": "x"}
    assert disp[1]["steps"][1]["result"] == "result A"


def test_display_history_multi_tool_call_turn(tmp_db):
    """Two tool calls in a single turn should produce two step pairs."""
    sid = "s4"
    turn = [
        {"role": "user", "content": "dual"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "recall", "arguments": '{"query":"1"}'}},
            {"id": "b", "type": "function",
             "function": {"name": "list_memories", "arguments": "{}"}},
        ]},
        {"role": "tool", "name": "recall", "tool_call_id": "a", "content": "r1"},
        {"role": "tool", "name": "list_memories", "tool_call_id": "b", "content": "r2"},
        {"role": "assistant", "content": "done"},
    ]
    db.append_turn(sid, "dual", turn)
    disp = db.get_display_history(sid)
    steps = disp[1]["steps"]
    types = [s["type"] for s in steps]
    assert types == ["tool_call", "tool_result", "tool_call", "tool_result"]
    assert steps[0]["name"] == "recall"
    assert steps[2]["name"] == "list_memories"


def test_sessions_preview_is_latest_user(tmp_db):
    sid = "s5"
    db.append_turn(sid, "older user line", _sample_turn("older user line"))
    db.append_turn(sid, "newest user preview text",
                   _sample_turn("newest user preview text"))
    rows = {s["session_id"]: s for s in db.get_sessions()}
    prev = rows[sid]["preview"]
    assert "newest" in prev
    assert "older" not in prev


def test_sessions_ordered_by_last_ts(tmp_db):
    db.append_turn("a", "first", _sample_turn("first"))
    db.append_turn("b", "second", _sample_turn("second"))
    sessions = db.get_sessions()
    # b was appended last so it should appear first (DESC by last_ts)
    assert sessions[0]["session_id"] == "b"


def test_clear_removes_session(tmp_db):
    sid = "s6"
    db.append_turn(sid, "hi", _sample_turn("hi"))
    assert db.get_history(sid)
    db.clear(sid)
    assert db.get_history(sid) == []


def test_legacy_append_function_removed():
    """`append()` was the legacy single-message insert; DESIGN §6.6 removed it."""
    assert not hasattr(db, "append"), "db.append() should be removed (legacy)"
