"""Fast unit tests for agent.py module-level helpers."""
from __future__ import annotations

import asyncio
import types

import pytest

import agent
import summarizer


# ── _visible_after_think ─────────────────────────────────────────────────────

@pytest.mark.parametrize("tag", ["thought", "think", "thinking", "redacted_reasoning", "redacted_thinking"])
def test_visible_after_think_strips_each_variant(tag):
    s = f"<{tag}>scratch</{tag}>\nHello"
    assert agent._visible_after_think(s) == "Hello"


def test_visible_after_think_strips_multiple_blocks():
    s = "<thought>a</thought>x<thinking>b</thinking>y"
    assert agent._visible_after_think(s) == "xy"


def test_visible_after_think_empty_when_only_block():
    assert agent._visible_after_think("<thought>only</thought>") == ""


def test_visible_after_think_noop_on_plain_text():
    assert agent._visible_after_think("Hi there.") == "Hi there."


def test_visible_after_think_handles_none():
    assert agent._visible_after_think(None) == ""


# Note: unclosed `<thought>` blocks are intentionally NOT pinned here. An
# unclosed tag is a model malfunction; the repair pass in the agent loop
# (see `test_repair_call_on_empty_visible`) is the actual recovery mechanism,
# so we don't make guarantees about what `_visible_after_think` returns in
# that case.


# ── _sanitize_history / _sanitize_message ────────────────────────────────────

def test_sanitize_drops_unknown_role():
    out = agent._sanitize_history([{"role": "function", "content": "x"}])
    assert out == []


def test_sanitize_keeps_tool_calls_shape():
    h = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "recall", "arguments": "{}"}},
        ],
    }]
    out = agent._sanitize_history(h)
    assert out[0]["tool_calls"][0]["id"] == "a"
    assert out[0]["tool_calls"][0]["function"]["name"] == "recall"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"
    # null content is preserved via absence (content key is conditionally set)
    assert "content" not in out[0] or out[0].get("content") is None


def test_sanitize_preserves_tool_message_name_and_id():
    h = [{"role": "tool", "name": "recall", "tool_call_id": "tc_1", "content": "res"}]
    out = agent._sanitize_history(h)
    assert out == [{"role": "tool", "name": "recall",
                    "tool_call_id": "tc_1", "content": "res"}]


def test_sanitize_tolerates_malformed_tool_call_entries():
    h = [{"role": "assistant", "tool_calls": ["nope", None, {"id": "x"}]}]
    out = agent._sanitize_history(h)
    tcs = out[0]["tool_calls"]
    # Only the dict-shaped entry should survive (gets filled with defaults)
    assert tcs == [{"id": "x", "type": "function",
                    "function": {"name": "", "arguments": ""}}]


# ── _extract_calls ───────────────────────────────────────────────────────────

def _msg_with_tc(name, args_json):
    fn = types.SimpleNamespace(name=name, arguments=args_json)
    tc = types.SimpleNamespace(function=fn, id="tc_1", type="function")
    return types.SimpleNamespace(tool_calls=[tc], content="")


def test_extract_calls_parses_valid_json():
    msg = _msg_with_tc("recall", '{"query": "zafir"}')
    assert agent._extract_calls(msg) == [{"name": "recall", "args": {"query": "zafir"}}]


def test_extract_calls_tolerates_bad_json():
    msg = _msg_with_tc("recall", "not json")
    assert agent._extract_calls(msg) == [{"name": "recall", "args": {}}]


def test_extract_calls_empty_when_no_tool_calls():
    msg = types.SimpleNamespace(content="hi", tool_calls=None)
    assert agent._extract_calls(msg) == []


# ── _build_system_prompt ─────────────────────────────────────────────────────

def test_build_system_prompt_includes_seeded_prompt_and_compact_tool_policy(tmp_system_prompt):
    prompt = agent._build_system_prompt()
    assert "Test system prompt" in prompt
    assert "at most one tool call per assistant message" in prompt
    assert "prefer apply_unified_patch" in prompt


def test_build_system_prompt_forbids_xml_tool_format(tmp_system_prompt):
    prompt = agent._build_system_prompt()
    # The generated tool docs section tells the model to use native tool_calls
    # only (not XML / fenced code).
    assert "tool_calls" in prompt


# ── _summarize_for_history fall-throughs (DESIGN §6.5) ───────────────────────

def _patch_summarize_gemma(monkeypatch, response_or_exc):
    """Replace agent.summarize_gemma (the Gemma 26B summarizer) with a stub.

    _summarize_for_history calls this via asyncio.to_thread — the stub is sync
    by design, matching the real signature.
    """
    def _fake(system_instruction, user_prompt, max_output_tokens=8192):
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc
    monkeypatch.setattr(agent, "summarize_gemma", _fake)


def test_summarize_for_history_returns_labelled_summary_on_success(monkeypatch):
    raw = "Z" * (agent._HISTORY_SUMMARIZE_THRESHOLD * 2)
    _patch_summarize_gemma(monkeypatch, "compact summary")

    result = asyncio.run(
        agent._summarize_for_history("recall", {}, "user msg", raw)
    )
    assert result.startswith("[history summary of recall]")
    assert "Takeaways:" in result
    assert "compact summary" in result
    assert len(result) < len(raw)


def test_summarize_for_history_truncates_when_llm_raises(monkeypatch):
    """If the Gemma 26B summarizer fails, fall back to truncating raw to
    _HISTORY_SUMMARIZE_THRESHOLD chars (DESIGN §6.5).
    """
    raw = "X" * (agent._HISTORY_SUMMARIZE_THRESHOLD * 2)
    _patch_summarize_gemma(monkeypatch, RuntimeError("summarizer unavailable"))

    result = asyncio.run(
        agent._summarize_for_history("recall", {}, "user msg", raw)
    )
    assert result == raw[: agent._HISTORY_SUMMARIZE_THRESHOLD]
    assert len(result) == agent._HISTORY_SUMMARIZE_THRESHOLD


def test_summarize_for_history_truncates_when_summary_is_empty(monkeypatch):
    """A summarizer reply that is entirely thinking-tags has zero visible
    content after stripping — treat as failure, fall back to truncation.
    """
    raw = "Y" * (agent._HISTORY_SUMMARIZE_THRESHOLD * 2)
    _patch_summarize_gemma(monkeypatch, "<thought>only thinking, no answer</thought>")

    result = asyncio.run(
        agent._summarize_for_history("recall", {}, "user msg", raw)
    )
    assert result == raw[: agent._HISTORY_SUMMARIZE_THRESHOLD]


def test_summarize_for_history_preserves_fetch_url_pagination_note(monkeypatch):
    """fetch_url's pagination footer must survive summarization — the agent
    needs `offset=…` to know the next page exists.
    """
    raw = (
        "long body " * 2000
        + "\n\n[… 5,000 more chars — call fetch_url with offset=8000 to continue]"
    )
    _patch_summarize_gemma(monkeypatch, "page 1 summary")

    result = asyncio.run(
        agent._summarize_for_history("fetch_url", {"url": "x"}, "user", raw)
    )
    assert "page 1 summary" in result
    assert "offset=8000" in result, f"pagination note dropped: {result!r}"


def test_summarize_for_history_does_not_add_pagination_to_other_tools(monkeypatch):
    """Pagination preservation is fetch_url-specific — the gate is the tool
    name, not the regex (other tools may legitimately mention offset numbers).
    """
    raw = "x" * 100 + "[… more — call fetch_url with offset=8000 to continue]"
    _patch_summarize_gemma(monkeypatch, "other tool summary")

    result = asyncio.run(
        agent._summarize_for_history("recall", {}, "user", raw)
    )
    assert "offset=8000" not in result


def test_summarize_for_history_uses_gemma_26b(monkeypatch):
    """§11.4 resolution: history summaries go through the dedicated 26B
    summarizer, NOT the primary provider chain. Assert summarize_gemma gets
    called and that _call is never touched.
    """
    raw = "W" * (agent._HISTORY_SUMMARIZE_THRESHOLD + 100)
    captured: dict = {}

    def _fake_gemma(system_instruction, user_prompt, max_output_tokens=8192):
        captured["system"] = system_instruction
        captured["user"] = user_prompt
        return "compact summary"
    monkeypatch.setattr(agent, "summarize_gemma", _fake_gemma)

    async def _boom(*args, **kwargs):
        raise AssertionError("_call must not be used by the history summarizer")
    monkeypatch.setattr(agent, "_call", _boom)

    result = asyncio.run(
        agent._summarize_for_history("recall", {"q": "x"}, "user msg", raw)
    )
    assert "compact summary" in result
    assert "Takeaways:" in result
    # Verify the prompt carried the structured tool metadata
    assert "Tool: recall" in captured["user"]
    assert "User request: user msg" in captured["user"]
    assert "TOOL OUTPUT:" in captured["user"]


# ── GEMINI_API_KEY_FREE_RESOLVED (summarizer HTTP only) ──────────────────────


class _FakeSummarizerHttpResponse:
    """Stand-in for httpx.Response from Gemini generateContent."""

    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}


class _FakeHttpxClient:
    """Captures the `key` query param from POST (summarizer API key)."""

    def __init__(self, captured: dict):
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, params=None, json=None):
        self._captured["key"] = (params or {}).get("key")
        return _FakeSummarizerHttpResponse()


def test_summarize_gemma_uses_gemini_api_key_free_resolved(monkeypatch):
    captured = {}
    monkeypatch.setattr(summarizer, "GEMINI_API_KEY_FREE_RESOLVED", "kfree")

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *a, **k: _FakeHttpxClient(captured))
    assert summarizer.summarize_gemma("s", "u") == "x"
    assert captured["key"] == "kfree"
