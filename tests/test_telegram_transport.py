"""Unit tests for Telegram transport (session mapping + command handling).

Hermetic: mocked httpx; no Telegram network.
"""
from __future__ import annotations

import asyncio
import json
import pytest

import config
import agent
import main
import telegram_transport as tt


@pytest.fixture(autouse=True)
def _clear_tg_state():
    tt._active_session.clear()
    yield
    tt._active_session.clear()


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"ok": True, "result": []}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeAsyncClient:
    """Minimal async client recording ``post`` / ``get`` calls."""

    def __init__(self):
        self.post_calls: list[tuple[str, dict | None]] = []
        self.get_calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url: str, json: dict | None = None, **kwargs):
        self.post_calls.append((url, json))
        if url.endswith('/sendMessage'):
            mid = len([u for u, _ in self.post_calls if u.endswith('/sendMessage')])
            return _FakeResponse(payload={"ok": True, "result": {"message_id": mid}})
        return _FakeResponse()

    async def get(self, url: str, params: dict | None = None, **kwargs):
        self.get_calls.append((url, dict(params or {})))
        return _FakeResponse()


def test_default_and_new_session_ids():
    assert tt._default_session_id(99, 0) == "tg:99"
    assert tt._default_session_id(99, 12) == "tg:99:12"
    sid = tt._new_session_id(99, 0)
    assert sid.startswith("tg:99:s-")
    assert len(sid.split(":")) == 3
    sl = sid.rsplit("s-", 1)[-1]
    assert len(sl) == tt._SESSION_SLUG_LEN
    assert sl.islower() and sl.isalpha() and sl.isascii()


def test_user_facing_label():
    assert tt._user_facing_label("tg:1") == "default"
    assert tt._user_facing_label("tg:1:s-abcde") == "abcde"


def test_resolve_session_switch_by_short_slug():
    valid = {"tg:1", "tg:1:s-abcde", "tg:1:s-xyzab"}
    assert tt._resolve_session_switch("tg:1:s-abcde", valid) == "tg:1:s-abcde"
    assert tt._resolve_session_switch("abcde", valid) == "tg:1:s-abcde"
    assert tt._resolve_session_switch("s-abcde", valid) == "tg:1:s-abcde"
    assert tt._resolve_session_switch("nope", valid) is None
    # two branches same slug (collision) — unresolvable as short
    v2 = {"tg:1:s-aaaaa", "tg:9:s-aaaaa"}
    assert tt._resolve_session_switch("aaaaa", v2) is None


def test_parse_command():
    assert tt._parse_command("hello") == (None, "hello")
    assert tt._parse_command("/new") == ("new", "")
    assert tt._parse_command("/switch tg:1:s-abc") == ("switch", "tg:1:s-abc")
    assert tt._parse_command("/help@MyBot args") == ("help", "args")


def test_active_session_per_chat_and_thread():
    assert tt._active_sid(1, 0) == "tg:1"
    assert tt._active_sid(1, 0) == "tg:1"
    tt._set_active(1, 0, "tg:1:s-abcde")
    assert tt._active_sid(1, 0) == "tg:1:s-abcde"
    # Different thread → different slot
    assert tt._active_sid(1, 7) == "tg:1:7"


def test_command_new_sets_active():
    client = FakeAsyncClient()

    async def _run():
        await tt._handle_command(client, "https://api.telegram.org/botTEST", 5, 0, "new", "")

    asyncio.run(_run())
    assert tt._active_sid(5, 0).startswith("tg:5:s-")
    sent = [c for c in client.post_calls if "sendMessage" in c[0]]
    assert sent
    p = sent[0][1] or {}
    assert "New session" in p.get("text", "")
    assert p.get("parse_mode") == "HTML"


def test_command_sessions_lists_tg_sessions_only(tmp_db, monkeypatch):
    monkeypatch.setattr(main, "_cache", {})

    async def fake_run(user_content, history, **kwargs):
        turn = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "ok"},
        ]
        return "ok", "fake", turn, []

    monkeypatch.setattr(agent, "run", fake_run)

    asyncio.run(main.complete_chat_turn("hi", "tg:9"))
    asyncio.run(main.complete_chat_turn("yo", "web-other"))

    client = FakeAsyncClient()

    async def _run():
        tt._set_active(9, 0, "tg:9")
        await tt._handle_command(client, "https://api.telegram.org/botTEST", 9, 0, "sessions", "")

    asyncio.run(_run())
    sent = [c[1]["text"] for c in client.post_calls if c[1] and "sendMessage" in c[0]]
    blob = "\n".join(sent)
    assert "default" in blob
    assert "web-other" not in blob


def test_command_switch_rejects_foreign_session(tmp_db, monkeypatch):
    monkeypatch.setattr(main, "_cache", {})

    async def fake_run(user_content, history, **kwargs):
        turn = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "x"},
        ]
        return "x", "fake", turn, []

    monkeypatch.setattr(agent, "run", fake_run)
    asyncio.run(main.complete_chat_turn("a", "tg:7:s-abcdef"))

    client = FakeAsyncClient()

    async def _run():
        tt._set_active(7, 0, "tg:7")
        await tt._handle_command(
            client, "https://api.telegram.org/botTEST", 7, 0, "switch", "tg:99:s-nope"
        )

    asyncio.run(_run())
    assert tt._active_sid(7, 0) == "tg:7"
    sent = [c[1]["text"] for c in client.post_calls if c[1] and "sendMessage" in c[0]]
    assert any("Unknown" in t and "ambiguous" in t for t in sent)


def test_command_switch_accepts_listed_session(tmp_db, monkeypatch):
    monkeypatch.setattr(main, "_cache", {})

    async def fake_run(user_content, history, **kwargs):
        turn = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "x"},
        ]
        return "x", "fake", turn, []

    monkeypatch.setattr(agent, "run", fake_run)
    sid = "tg:3:s-zzzzz"
    asyncio.run(main.complete_chat_turn("x", sid))

    client = FakeAsyncClient()

    async def _run():
        tt._set_active(3, 0, "tg:3")
        # Short slug only (no digits)
        await tt._handle_command(client, "https://api.telegram.org/botTEST", 3, 0, "switch", "zzzzz")

    asyncio.run(_run())
    assert tt._active_sid(3, 0) == sid


def test_allowed_user_ids_blocks_stranger(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", frozenset({100}))
    calls: list[str] = []

    async def fake_complete(msg: str, sid: str, *, attachments=None, **kwargs):
        calls.append(msg)
        return "no", "fake"

    monkeypatch.setattr(main, "complete_chat_turn", fake_complete)
    client = FakeAsyncClient()

    async def _run():
        await tt._handle_message(
            client,
            "https://api.telegram.org/botTEST",
            {"chat": {"id": 8}, "from": {"id": 999}, "text": "hello"},
        )

    asyncio.run(_run())
    assert calls == []


def test_allowed_user_ids_allows_listed_user(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", frozenset({100}))
    calls: list[tuple[str, str]] = []

    async def fake_run_stream_turn(state, user_content, history, *, output_channel="default"):
        calls.append((state.session_id, user_content))
        state.full_response = "ok"
        await state.queue.put({"type": "text_chunk", "text": "ok"})
        state.completed = True
        await state.queue.put(None)

    monkeypatch.setattr(main, "_run_stream_turn", fake_run_stream_turn)
    client = FakeAsyncClient()

    async def _run():
        await tt._handle_message(
            client,
            "https://api.telegram.org/botTEST",
            {"chat": {"id": 8}, "from": {"id": 100}, "text": "hello"},
        )

    asyncio.run(_run())
    assert calls == [("tg:8", "hello")]


def test_partial_stream_edit_and_sanitization(monkeypatch):
    # Disable allowlist so the test is hermetic regardless of local .env.
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", None)

    async def fake_run_stream_turn(state, user_content, history, *, output_channel="default"):
        assert output_channel == "telegram"
        # Simulate leaked think tags and markdown bold in model text.
        state.full_response = "<thinking>private</thinking>hi **there**"
        await state.queue.put({"type": "text_chunk", "text": "hi **there**"})
        state.completed = True
        await state.queue.put(None)

    monkeypatch.setattr(main, "_run_stream_turn", fake_run_stream_turn)
    client = FakeAsyncClient()

    async def _run():
        await tt._handle_message(
            client,
            "https://api.telegram.org/botTEST",
            {"chat": {"id": 8}, "from": {"id": 42}, "text": "hello tg"},
        )

    asyncio.run(_run())
    edits = [body for url, body in client.post_calls if url.endswith("/editMessageText")]
    assert edits, "expected editMessageText calls for partial streaming"
    assert any("hi there" in (e.get("text") or "") for e in edits)
    assert all("**" not in (e.get("text") or "") for e in edits)
    assert all("<thinking>" not in (e.get("text") or "") for e in edits)


def test_agent_reply_for_tg_strips_think_and_double_asterisk():
    out = tt._agent_reply_for_tg("<thinking>x</thinking>Hello **bold**")
    assert "thinking" not in out.lower()
    assert "**" not in out
    assert "Hello bold" in out
