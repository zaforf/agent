"""FastAPI endpoint tests with mocked agent so no LLM calls fire.

Exercise:
- /health
- /sessions + /sessions/{id}/history + DELETE /sessions/{id}
- /chat (non-streaming): persists turn_messages + updates cache
- /chat/stream (SSE): emits events + persists on done
- The in-memory _cache short-circuits repeat DB reads
"""
from __future__ import annotations

import asyncio
import json
import threading
import pytest
from fastapi.testclient import TestClient

import main
import db
import agent


@pytest.fixture
def client(tmp_db, monkeypatch):
    # Isolate the per-process cache between tests
    monkeypatch.setattr(main, "_cache", {})
    with TestClient(main.app) as c:
        yield c


# ── Health & basic endpoints ──────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "providers" in body
    assert "web_search_configured" in body


def test_sessions_empty(client):
    r = client.get("/sessions")
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


def test_history_empty_for_unknown_session(client):
    r = client.get("/sessions/unknown-id/history")
    assert r.status_code == 200
    assert r.json() == {"messages": [], "active": False}


# ── /chat (non-streaming) ────────────────────────────────────────────────────

def test_chat_persists_turn_messages(client, monkeypatch):
    async def fake_run(user_message, history, **kwargs):
        turn = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "from-fake"},
        ]
        return "from-fake", "fake-provider", turn, []

    monkeypatch.setattr(agent, "run", fake_run)

    r = client.post("/chat", json={"message": "hi", "session_id": "s1"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "from-fake"
    assert body["provider"] == "fake-provider"

    # Persisted in DB
    hist = db.get_history("s1")
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[1]["content"] == "from-fake"

    # And visible on display endpoint
    r2 = client.get("/sessions/s1/history")
    msgs = r2.json()["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "from-fake"


def test_chat_uses_cache_on_second_request(client, monkeypatch):
    """Second request for the same session should not re-read the DB."""
    async def fake_run(user_message, history, **kwargs):
        return "ok", "p", [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "ok"},
        ], []
    monkeypatch.setattr(agent, "run", fake_run)

    reads = []
    orig = db.get_history
    def _spy(sid):
        reads.append(sid)
        return orig(sid)
    monkeypatch.setattr(db, "get_history", _spy)

    client.post("/chat", json={"message": "one", "session_id": "c1"})
    client.post("/chat", json={"message": "two", "session_id": "c1"})

    assert reads == ["c1"], f"DB should be read exactly once per session, got {reads}"


def test_chat_non_blocking_summary_patches_db_row(tmp_db, monkeypatch):
    """DESIGN §6.5 end-to-end: when agent.run returns pending summary tasks,
    main.py must (a) return the response to the user immediately with raw
    content stored, (b) drain the pending tasks in the background, (c)
    overwrite the stored turn with the summary once the tasks finish.

    Calls `main.chat()` directly instead of going through TestClient so the
    whole scenario shares one event loop — pending tasks, Events, and the
    background finalizer all need to be on the same loop.
    """
    import asyncio

    import main as main_module

    async def _scenario():
        # Isolate per-test state in the shared main module.
        monkeypatch.setattr(main_module, "_cache", {})
        monkeypatch.setattr(main_module, "_background_tasks", set())

        release = asyncio.Event()

        async def _delayed_summary():
            await release.wait()
            return "[history summary of big_tool]\nDELAYED_SUMMARY"

        async def fake_run(user_message, history, **kwargs):
            tool_msg = {
                "role": "tool",
                "name": "big_tool",
                "tool_call_id": "tc_1",
                "content": "X" * 9000,
            }
            turn = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "tc_1", "type": "function",
                                 "function": {"name": "big_tool", "arguments": "{}"}}]},
                tool_msg,
                {"role": "assistant", "content": "done"},
            ]
            pending = [(tool_msg, asyncio.create_task(_delayed_summary()))]
            return "done", "fake", turn, pending

        monkeypatch.setattr(agent, "run", fake_run)

        # 1. Handler returns immediately — summary still in flight.
        req = main_module.ChatRequest(message="hi", session_id="bg1")
        resp = await main_module.chat(req)
        assert resp.response == "done"
        assert any(
            "_finalize_summaries" in str(t.get_coro()) for t in main_module._background_tasks
        ), "finalizer task should be registered and still running"

        # 2. DB row holds the raw content at this point.
        hist_before = db.get_history("bg1")
        tool_row_before = [m for m in hist_before if m["role"] == "tool"][0]
        assert tool_row_before["content"] == "X" * 9000, (
            "DB should initially hold raw content — summary is still pending"
        )

        # 3. Release the summary and wait for the finalizer to fire.
        release.set()
        for _ in range(100):
            await asyncio.sleep(0)
            if not main_module._background_tasks:
                break
        else:
            raise AssertionError("background finalizer never completed")

        # 4. DB row was patched.
        hist_after = db.get_history("bg1")
        tool_row_after = [m for m in hist_after if m["role"] == "tool"][0]
        assert "DELAYED_SUMMARY" in tool_row_after["content"], (
            f"DB row was not updated with the summary: "
            f"{tool_row_after['content'][:100]!r}"
        )

    asyncio.run(_scenario())




def test_chat_nuke_resets_history_to_assistant_summary(client, monkeypatch):
    # Seed prior history so we can verify full reset.
    db.append_turn("n1", "old-user", [
        {"role": "user", "content": "old-user"},
        {"role": "assistant", "content": "old-answer"},
    ])

    async def fake_run(user_message, history, **kwargs):
        return "summary kept", "fake-provider", [
            {"role": "assistant", "content": "summary kept", "_nuke": True},
        ], []

    monkeypatch.setattr(agent, "run", fake_run)

    r = client.post("/chat", json={"message": "please nuke", "session_id": "n1"})
    assert r.status_code == 200
    expected = "Chat reset via nuke. Summary:\nsummary kept"
    assert r.json()["response"] == expected

    hist = db.get_history("n1")
    assert hist == [{"role": "assistant", "content": expected}]

    shown = client.get("/sessions/n1/history").json()["messages"]
    assert shown == [{"role": "assistant", "content": expected}]

def test_chat_propagates_agent_error(client, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("provider exhausted")
    monkeypatch.setattr(agent, "run", boom)

    r = client.post("/chat", json={"message": "x", "session_id": "err1"})
    assert r.status_code == 500
    assert "provider exhausted" in r.json()["detail"]


# ── /chat/stream (SSE) ───────────────────────────────────────────────────────

def test_chat_stream_events_and_persist(client, monkeypatch):
    async def fake_stream(user_message, history, **kwargs):
        yield {"type": "text_chunk", "text": "hel"}
        yield {"type": "text_chunk", "text": "lo"}
        yield {"type": "done", "provider": "fake",
               "turn_messages": [
                   {"role": "user", "content": user_message},
                   {"role": "assistant", "content": "hello"},
               ]}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream("POST", "/chat/stream",
                       json={"message": "go", "session_id": "s-stream"}) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]

    events = [json.loads(ln[6:]) for ln in lines]
    types = [e["type"] for e in events]
    assert types == ["status", "text_chunk", "text_chunk", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "text_chunk") == "hello"

    # Persisted
    hist = db.get_history("s-stream")
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[1]["content"] == "hello"


def test_chat_stream_handles_error_event(client, monkeypatch):
    async def fake_stream(user_message, history, **kwargs):
        yield {"type": "text_chunk", "text": "partial"}
        raise RuntimeError("boom midway")

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream("POST", "/chat/stream",
                       json={"message": "x", "session_id": "err-stream"}) as r:
        data = b"".join(r.iter_bytes()).decode()

    assert "boom midway" in data
    hist = db.get_history("err-stream")
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[0]["content"] == "x"
    assert "partial" in (hist[1].get("content") or "")
    assert "boom midway" in (hist[1].get("content") or "")


def test_chat_stream_persists_on_agent_error_yield(client, monkeypatch):
    async def fake_stream(user_message, history, **kwargs):
        yield {"type": "text_chunk", "text": "hi"}
        yield {"type": "error", "detail": "max tool iterations"}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "ask", "session_id": "err-yield"},
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()

    assert '"type": "error"' in body
    hist = db.get_history("err-yield")
    assert len(hist) == 2
    assert hist[1]["content"].startswith("hi")
    assert "max tool iterations" in hist[1]["content"]


def test_chat_stream_cancel_endpoint_cancels_active_stream(client, monkeypatch):
    started = threading.Event()

    async def fake_stream(user_message, history, **kwargs):
        started.set()
        yield {"type": "text_chunk", "text": "partial"}
        while True:
            await asyncio.sleep(1)

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    stream_data = {}

    def _consume_stream():
        with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "x", "session_id": "cancel-sess"},
        ) as r:
            stream_data["status"] = r.status_code
            stream_data["body"] = b"".join(r.iter_bytes()).decode()

    t = threading.Thread(target=_consume_stream, daemon=True)
    t.start()

    assert started.wait(timeout=2), "stream never started"
    rc = client.post("/chat/stream/cancel", json={"session_id": "cancel-sess"})
    assert rc.status_code == 200
    assert rc.json() == {"cancelled": True}

    t.join(timeout=5)
    assert not t.is_alive(), "stream thread should finish after cancellation"
    assert stream_data.get("status") == 200
    assert '"type": "cancelled"' in stream_data.get("body", "")
    # No history should be persisted on cancel
    assert db.get_history("cancel-sess") == []


def test_chat_stream_error_uses_turn_messages_when_sent(client, monkeypatch):
    """Error payload may include full turn (e.g. tool_calls + tool + error footer)."""
    tc = {
        "id": "tc_0",
        "type": "function",
        "function": {"name": "workspace_read", "arguments": "{}"},
    }
    tm = [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {
            "role": "tool",
            "name": "workspace_read",
            "tool_call_id": "tc_0",
            "content": "file contents",
        },
        {"role": "assistant", "content": "**Error:** injected"},
    ]

    async def fake_stream(user_message, history, **kwargs):
        assert user_message == "ask"
        yield {"type": "error", "detail": "injected", "turn_messages": tm}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "ask", "session_id": "err-tm"},
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()
    assert "turn_messages" not in body
    hist = db.get_history("err-tm")
    assert [m.get("role") for m in hist] == ["user", "assistant", "tool", "assistant"]
    assert hist[2]["name"] == "workspace_read"
    assert hist[2]["content"] == "file contents"
    r = client.post("/chat/stream/cancel", json={"session_id": "nope"})
    assert r.status_code == 200
    assert r.json() == {"cancelled": False}


def test_chat_stream_cancel_event_does_not_persist_history(client, monkeypatch):
    """If run_stream emits a cancelled event itself, no turn should persist."""
    async def fake_stream(user_message, history, **kwargs):
        yield {"type": "text_chunk", "text": "partial"}
        yield {"type": "cancelled"}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "x", "session_id": "cancel-no-persist"},
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()

    assert '"type": "cancelled"' in body
    # No history should be persisted on cancel
    assert db.get_history("cancel-no-persist") == []


def test_stream_turn_background_persists_without_sse_consumer(tmp_db, monkeypatch):
    """Server-owned producer persists turn even without an active SSE consumer."""
    monkeypatch.setattr(main, "_cache", {})

    async def fake_stream(user_message, history, **kwargs):
        yield {"type": "text_chunk", "text": "partial"}
        yield {
            "type": "done",
            "provider": "fake",
            "turn_messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "final-result"},
            ],
        }

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    async def _scenario():
        # Mock the immediate persistence that happens in the API endpoint
        row_id = db.create_pending_turn("dc1", "persist me")
        state = main._StreamTurnState("dc1", "persist me", [], row_id)
        history = main._get_history("dc1")
        await main._run_stream_turn(state, "persist me", history)

    asyncio.run(_scenario())

    hist = db.get_history("dc1")
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[-1]["content"] == "final-result"


# ── DELETE /sessions/{id} ────────────────────────────────────────────────────

def test_delete_session_clears_cache_and_db(client, monkeypatch):
    async def fake_run(msg, h, **kwargs):
        return "ok", "p", [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": "ok"},
        ], []
    monkeypatch.setattr(agent, "run", fake_run)

    client.post("/chat", json={"message": "x", "session_id": "dl"})
    assert db.get_history("dl")
    assert "dl" in main._cache

    r = client.delete("/sessions/dl")
    assert r.status_code == 200
    assert "dl" not in main._cache
    assert db.get_history("dl") == []
