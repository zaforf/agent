"""Shared pytest fixtures.

Kept dependency-free and model-free so `pytest` (fast default) never touches
the network, the production SQLite DB, or the real system prompt file.

Async tests use plain `asyncio.run(...)` inside sync test functions so we avoid
a `pytest-asyncio` dependency.
"""
from __future__ import annotations

import json as _json
import sys
from pathlib import Path

import pytest

# Make project root importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Filesystem fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point db.DB_PATH at an empty tmp file and run init()."""
    import db

    db_file = tmp_path / "history.db"
    monkeypatch.setattr(db, "DB_PATH", db_file)
    db.init()
    return db_file


@pytest.fixture
def tmp_system_prompt(tmp_path, monkeypatch):
    """Point config.SYSTEM_PROMPT_PATH at a tmp file seeded with a known marker."""
    import config

    sp = tmp_path / "system_prompt.md"
    sp.write_text("# Test system prompt\nHello.\n")
    monkeypatch.setattr(config, "SYSTEM_PROMPT_PATH", sp)
    return sp


# ── Fake OpenAI-compatible response objects ──────────────────────────────────

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, choices):
        self.choices = choices


def make_response(content="", tool_calls=None):
    """Build a non-streaming fake response.

    tool_calls: list of {"id", "name", "args"} dicts.
    """
    tcs = []
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tcs.append(_ToolCall(
                id=tc.get("id", f"tc_{i}"),
                name=tc["name"],
                arguments=_json.dumps(tc.get("args", {})),
            ))
    return _Response([_Choice(_Message(content, tcs))])


# Streaming chunk objects
class _DeltaFn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _DeltaToolCall:
    def __init__(self, index, id, name, arguments):
        self.index = index
        self.id = id
        self.type = "function"
        self.function = _DeltaFn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StreamChoice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, choices):
        self.choices = choices


def text_chunk(text):
    return _Chunk([_StreamChoice(_Delta(content=text))])


def tool_chunk(index, id, name, arguments):
    return _Chunk([_StreamChoice(_Delta(tool_calls=[
        _DeltaToolCall(index=index, id=id, name=name, arguments=arguments)
    ]))])


# ── Scripted provider fixtures ────────────────────────────────────────────────

class _FakeCompletions:
    """Records every .create() call and returns scripted responses in order.

    Each scripted response is either:
      - a `_Response` (non-streaming)
      - a list of `_Chunk` (streaming, consumed as `stream=True`)
      - an Exception (raised on that call)
    """

    def __init__(self, responses, name):
        self._responses = list(responses)
        self.calls = []
        self.name = name

    async def create(self, *, model, messages, max_tokens=8192, tools=None,
                     tool_choice=None, stream=False, **kwargs):
        rec = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": tools,
            "stream": stream,
        }
        if "parallel_tool_calls" in kwargs:
            rec["parallel_tool_calls"] = kwargs["parallel_tool_calls"]
        self.calls.append(rec)
        if not self._responses:
            raise RuntimeError(f"FakeProvider {self.name!r} script exhausted")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if stream:
            if not isinstance(nxt, list):
                raise TypeError(f"Streaming call expected a list of chunks; got {type(nxt)}")
            async def _gen():
                for c in nxt:
                    yield c
            return _gen()
        return nxt


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("_Chat", (), {"completions": completions})()


def install_providers(monkeypatch, chains, names=None):
    """Install a provider chain on agent._clients.

    `chains` is a list of scripts — one per provider. Each script is a list of
    responses (see `_FakeCompletions`).

    Returns the single `_FakeCompletions` when only one provider is installed,
    otherwise a list (so callers can assert on multiple provider histories).
    """
    import agent
    if names is None:
        names = [f"fake-{i}" for i in range(len(chains))]
    entries = []
    completion_objs = []
    for nm, script in zip(names, chains):
        comp = _FakeCompletions(script, nm)
        completion_objs.append(comp)
        entries.append({"name": nm, "model": f"m-{nm}", "client": _FakeClient(comp)})
    monkeypatch.setattr(agent, "_clients", entries)
    if len(completion_objs) == 1:
        return completion_objs[0]
    return completion_objs


@pytest.fixture
def providers(monkeypatch):
    """Fixture wrapper around install_providers for convenience."""
    def _install(chains, names=None):
        return install_providers(monkeypatch, chains, names=names)
    return _install


@pytest.fixture(autouse=True)
def _stub_title_summarizer(monkeypatch):
    """Prevent session-title generation from making real Gemini API calls in unit tests."""
    try:
        import main as _main
        monkeypatch.setattr(_main, "summarize_gemma", lambda *a, **kw: "Stub Title")
    except Exception:
        pass
