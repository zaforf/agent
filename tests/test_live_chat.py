"""Live end-to-end tests hitting the real provider chain.

These are marked `@pytest.mark.live` so the default `pytest` run skips them.
Run with `pytest -m live` when you want to exercise the real model.

Kept to 4 tests total to respect rate limits (Gemini free tier is 15 rpm).
"""
from __future__ import annotations

import asyncio

import pytest


pytestmark = pytest.mark.live


def test_simple_response():
    """Smallest possible round-trip — confirms the provider chain is wired."""
    import agent
    resp, _provider, _turn, _pending = asyncio.run(
        agent.run("Reply with exactly: PING_OK", [])
    )
    assert "PING_OK" in resp, f"got: {resp!r}"


def test_history_round_trip_purely_from_context():
    """Turn 1 establishes a fact; turn 2 must recall it from conversation
    history alone (no memory tool). This exercises the core design promise
    that history is fed back into every call.
    """
    import agent
    _, _, turn1, _ = asyncio.run(
        agent.run(
            "Remember this codeword for the next message: SAPPHIRE-42. "
            "Acknowledge with exactly: OK",
            [],
        )
    )
    resp2, _, _, _ = asyncio.run(
        agent.run(
            "What was the codeword I just gave you? Reply with only the codeword.",
            turn1,
        )
    )
    assert "SAPPHIRE-42" in resp2, f"history-dependent recall failed: {resp2!r}"


def test_streaming_yields_multiple_chunks_and_done():
    """Streaming path actually streams — assert >1 text_chunk + a final done event.

    Asks for a long-enough response (~300 chars) that any sane provider must
    emit at least two SSE chunks; otherwise the test wouldn't really verify
    streaming, just request/response.
    """
    import agent

    prompt = (
        "Write a 250-300 character paragraph explaining what HTTP is in plain "
        "language. End the paragraph with the literal token STREAM_OK on its "
        "own. No markdown."
    )

    async def _collect():
        out = []
        async for ev in agent.run_stream(prompt, []):
            out.append(ev)
        return out

    events = asyncio.run(_collect())
    types = [e["type"] for e in events]
    text_chunks = [e for e in events if e["type"] == "text_chunk"]
    text = "".join(e["text"] for e in text_chunks)

    assert "done" in types, f"no done event: {types}"
    assert "STREAM_OK" in text, f"marker missing in response: {text!r}"
    assert len(text_chunks) >= 2, (
        f"expected multiple chunks (real streaming) but got {len(text_chunks)}; "
        f"text was {text!r}"
    )


@pytest.mark.net  # also hits Wikipedia on top of the live LLM
def test_fetch_and_summarize_end_to_end(monkeypatch):
    """End-to-end integration test — the agent must:

      1. Call fetch_url on a real URL (Wikipedia)
      2. tools/web.py's internal summarizer must return a real result, not
         raise-and-fall-through-to-raw
      3. The final visible response must be non-empty and on-topic

    This pins the exact regression shape we saw when a deprecated summarizer
    model silently broke: the summarizer call would raise, fetch_url would
    silently fall back to raw text, and the agent would produce a low-quality
    or empty answer. None of that surfaces in unit tests because the
    summarizer is mocked there.
    """
    import agent
    from tools import web as fetch

    summarizer_results: list[str] = []
    real_summarize = fetch._summarize_content

    def spy(text, prompt):
        result = real_summarize(text, prompt)
        summarizer_results.append(result)
        return result

    monkeypatch.setattr(fetch, "_summarize_content", spy)

    response, _provider, turn, _pending = asyncio.run(agent.run(
        "Fetch https://en.wikipedia.org/wiki/Twice and list a few of their "
        "studio albums. A short bulleted list is fine.",
        [],
    ))

    # 1. fetch_url was actually invoked
    fetch_calls = [m for m in turn if m.get("role") == "tool" and m.get("name") == "fetch_url"]
    assert fetch_calls, (
        f"expected fetch_url to be invoked; got roles={[m['role'] for m in turn]}"
    )

    # 2. The summarizer succeeded at least once. This is the deprecated-model
    # sentinel — on regression the list is empty because _summarize_content
    # raised and was swallowed by fetch_url's try/except.
    assert summarizer_results, (
        "summarizer was never called successfully — either fetch_url was skipped "
        "or the summarizer raised and fell back to raw text (common symptom of "
        "a deprecated summarizer model)"
    )
    assert all(isinstance(r, str) and r.strip() for r in summarizer_results), (
        f"summarizer returned empty results: {summarizer_results!r}"
    )

    # 3. Final response is meaningful and on-topic
    assert len(response) > 50, f"response too short: {response!r}"
    assert "album" in response.lower() or "twice" in response.lower(), (
        f"response doesn't look on-topic: {response!r}"
    )


def test_tool_call_trace_in_turn_messages():
    """Force a tool call via list_memories and verify tool trace shape."""
    import agent
    _, _, turn, _ = asyncio.run(
        agent.run(
            "Call list_memories, then reply in one short sentence.",
            [],
        )
    )
    roles = [m["role"] for m in turn]
    assert roles[0] == "user"
    assert "tool" in roles, f"no tool message emitted: {roles}"
    assert roles[-1] == "assistant"
    assert turn[-1].get("content"), "final assistant has empty content"
