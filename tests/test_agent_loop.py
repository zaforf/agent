"""Agentic-loop invariants — the core design contract.

These tests mock the provider chain and tool execution so the whole agent loop
can be exercised in <100ms each, with no network. They cover:

- Tool-call results flow back into the next LLM call (DESIGN §4.1)
- The summarization-timing invariant (DESIGN §4.1, §6.5)
- Repair call when the model's visible response is empty (DESIGN §4.3)
- Multi-iteration tool loops
- MAX_TOOL_ITERATIONS termination
- Provider-chain fallback on retryable errors (DESIGN §3)
- Streaming: tool-call events + text_chunk + done event ordering
- Streaming repair triggers correctly when the model returns only thinking
  tokens (pinned by test_streaming_repair_triggers_on_thinking_only)
"""
from __future__ import annotations

import asyncio

import pytest

import agent
from tests.conftest import make_response, text_chunk, tool_chunk


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _noop_sleep(_):
    """Drop-in for asyncio.sleep to kill backoff time in tests."""
    return None


def _install_fake_tool(monkeypatch, name, impl):
    """Register a fake callable as a tool — overrides the real registry."""
    monkeypatch.setattr(agent, "TOOL_FUNCTIONS", {name: impl})


# ── 1. Tool results flow back into next LLM call ─────────────────────────────

def test_tool_results_flow_back_into_next_llm_call(monkeypatch, tmp_system_prompt, providers):
    """Iteration 2's message list must contain the exact tool result from iter 1."""
    _install_fake_tool(monkeypatch, "fake_tool", lambda: "TOOL_RESULT_SENTINEL")
    script = [
        make_response(content="", tool_calls=[{"id": "tc_1", "name": "fake_tool"}]),
        make_response(content="final answer"),
    ]
    p = providers([script])

    response, provider, turn, _ = asyncio.run(agent.run("do it", []))

    assert response == "final answer"
    assert provider == "fake-0"
    # Two provider calls — one per iteration
    assert len(p.calls) == 2
    # Iteration 2's messages must include the tool result
    call2_msgs = p.calls[1]["messages"]
    tool_msgs = [m for m in call2_msgs if m.get("role") == "tool"]
    assert tool_msgs, f"no tool message in call 2: {call2_msgs}"
    assert tool_msgs[0]["content"] == "TOOL_RESULT_SENTINEL"
    # tool_call_id must match the id the model emitted
    assert tool_msgs[0]["tool_call_id"] == "tc_1"


def test_turn_messages_contain_full_tool_trace(monkeypatch, tmp_system_prompt, providers):
    """turn_messages must record user → assistant(tool_calls) → tool → assistant."""
    _install_fake_tool(monkeypatch, "fake_tool", lambda: "result X")
    providers([[
        make_response(content="", tool_calls=[{"id": "tc_1", "name": "fake_tool"}]),
        make_response(content="done"),
    ]])

    _, _, turn, _ = asyncio.run(agent.run("go", []))
    roles = [m["role"] for m in turn]
    assert roles == ["user", "assistant", "tool", "assistant"], roles
    assert turn[1].get("tool_calls"), "intermediate assistant must carry tool_calls"
    assert turn[2]["name"] == "fake_tool"
    assert turn[2]["content"] == "result X"
    assert turn[3]["content"] == "done"


# ── 2. Summarization-timing invariant (DESIGN §4.1) ──────────────────────────

def test_summarization_runs_after_llm_sees_raw_output(monkeypatch, tmp_system_prompt, providers):
    """The CRITICAL invariant from DESIGN §4.1:
    - The model responding to a tool call must see the FULL tool output.
    - Summarization replaces content at turn end (and in later replay), not
      before the next in-turn LLM call.
    """
    raw = "X" * (agent._HISTORY_SUMMARIZE_THRESHOLD + 500)
    _install_fake_tool(monkeypatch, "big_tool", lambda: raw)

    async def _fake_summary(name, args, user_message, content):
        return "<SUMMARY>"
    monkeypatch.setattr(agent, "_summarize_for_history", _fake_summary)

    p = providers([[
        make_response(content="", tool_calls=[{"id": "tc_1", "name": "big_tool"}]),
        make_response(content="final"),
    ]])

    _, _, turn, _ = asyncio.run(agent.run("go", []))

    # At the time of the SECOND LLM call, the tool message must still carry
    # the raw output — the model responding to big_tool needs full data.
    call2_msgs = p.calls[1]["messages"]
    tool_msg_at_call2 = [m for m in call2_msgs if m.get("role") == "tool"][0]
    assert tool_msg_at_call2["content"] == raw, (
        "model responding to the tool must see RAW output, not the summary"
    )

    # After run() returns, turn_messages reflects post-turn-end resolve —
    # the tool message has been replaced with the summary for storage / replay.
    tool_msg_final = [m for m in turn if m.get("role") == "tool"][0]
    assert tool_msg_final["content"] == "<SUMMARY>"


def test_multi_iteration_tools_prior_results_stay_raw_until_turn_end(
    monkeypatch, tmp_system_prompt, providers
):
    """DESIGN §4.1: With tool A then tool B in one user turn, the LLM call that
    runs after B is appended must still see A's full raw output. Summarization
    must not swap A to a history summary mid-turn — only after the final
    no-tools reply (so the model can chain reasoning on all raw tool data).
    """
    raw_a = "A" * (agent._HISTORY_SUMMARIZE_THRESHOLD + 500)

    def _tool_a():
        return raw_a

    monkeypatch.setattr(
        agent,
        "TOOL_FUNCTIONS",
        {"tool_a": _tool_a, "tool_b": lambda: "B_RESULT"},
    )

    async def _instant_hist(name, args, user_message, content):
        """Avoid thread timing — turn-end apply must see task.done() after sleep(0)."""
        return (
            f"[history summary of {name}]\n"
            "This tool response was summarized for context efficiency. Takeaways:\n"
            "COMPACT_A"
        )

    monkeypatch.setattr(agent, "_summarize_for_history", _instant_hist)

    p = providers([[
        make_response(content="", tool_calls=[{"id": "t1", "name": "tool_a"}]),
        make_response(content="", tool_calls=[{"id": "t2", "name": "tool_b"}]),
        make_response(content="done"),
    ]])

    _, _, turn, _ = asyncio.run(agent.run("chain", []))

    assert len(p.calls) == 3
    call3_msgs = p.calls[2]["messages"]
    tool_msgs = [m for m in call3_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, tool_msgs
    assert tool_msgs[0]["content"] == raw_a, (
        "call after tool_b must still see full tool_a — not summarized mid-turn"
    )
    assert tool_msgs[1]["content"] == "B_RESULT"

    tool_a_row = [m for m in turn if m.get("role") == "tool" and m.get("name") == "tool_a"][0]
    assert tool_a_row["content"].startswith("[history summary of tool_a]")
    assert "COMPACT_A" in tool_a_row["content"]


def test_raw_tool_content_preserved_even_when_summary_wins_race(
    monkeypatch, tmp_system_prompt, providers
):
    """§4.1 ordering guard: `_apply_finished_summaries` MUST run after
    `_call`, never before. That ordering is the only thing preventing the
    summary from overwriting `msg_dict["content"]` before the request
    carrying the raw content is sent to the API.

    Once the request is out, the summarizer finishing is harmless — the
    server already read the message list. So the test specifically probes
    the pre-send window: force the summary to be `done()` the moment iter
    2's LLM call opens, then check the live tool message still holds raw
    content. If a future refactor moves `_apply_finished_summaries` ahead
    of `_call`, or makes the summary task mutate the dict directly, this
    test fails before anyone ships the regression.
    """
    raw = "X" * (agent._HISTORY_SUMMARIZE_THRESHOLD + 500)
    _install_fake_tool(monkeypatch, "big_tool", lambda: raw)

    summary_done = asyncio.Event()

    async def _fast_summary(name, args, user_message, content):
        summary_done.set()
        return "<SUMMARY>"
    monkeypatch.setattr(agent, "_summarize_for_history", _fast_summary)

    p = providers([[
        make_response(content="", tool_calls=[{"id": "tc_1", "name": "big_tool"}]),
        make_response(content="final"),
    ]])

    # Wrap p.create so iter 2 blocks until the summary task is done.
    # The summary genuinely wins the race — task.done() is True before the
    # LLM call returns — yet the invariant must hold.
    orig_create = p.create
    captured_during_call: list[str] = []

    async def racing_create(**kwargs):
        call_idx = len(p.calls)  # 0 before iter-1's record, 1 before iter-2's
        if call_idx == 1:
            await summary_done.wait()
            # Summary task is done; check live state (not the snapshot copy)
            # of the tool message the LLM is about to reason about.
            tool_msg = [m for m in kwargs["messages"] if m.get("role") == "tool"][0]
            captured_during_call.append(tool_msg["content"])
        return await orig_create(**kwargs)

    monkeypatch.setattr(p, "create", racing_create)

    _, _, turn, _ = asyncio.run(agent.run("go", []))

    # The summary finished before iter 2's LLM call returned (we waited on it).
    assert summary_done.is_set()
    # Yet at the moment the LLM was about to respond, the live tool message
    # still held raw content — not the already-computed summary.
    assert captured_during_call == [raw], (
        f"§4.1 invariant violated: summary mutated the tool message before "
        f"the LLM responded to it. content[:60]="
        f"{(captured_during_call or ['<none>'])[0][:60]!r}"
    )
    # Sanity: the _FakeCompletions snapshot (shallow-copied at call entry)
    # agrees — the model's request carried raw content.
    call2_tool = [m for m in p.calls[1]["messages"] if m.get("role") == "tool"][0]
    assert call2_tool["content"] == raw

    # Post-turn: once the LLM has responded, the stored form carries the
    # summary so subsequent turns don't re-send the raw blob.
    tool_msg_final = [m for m in turn if m.get("role") == "tool"][0]
    assert tool_msg_final["content"] == "<SUMMARY>"


def test_history_summarization_is_non_blocking(monkeypatch, tmp_system_prompt, providers):
    """DESIGN §6.5 non-blocking guarantee: a slow summary MUST NOT stall a
    turn. The user gets their response immediately; the unfinished summary
    task is handed back in the `pending_summaries` return value for the
    caller to drain in the background.
    """
    raw = "X" * (agent._HISTORY_SUMMARIZE_THRESHOLD + 500)
    _install_fake_tool(monkeypatch, "big_tool", lambda: raw)

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_summary(name, args, user_message, content):
        started.set()
        await release.wait()         # blocks until the test explicitly releases
        return f"[history summary of {name}]\nDELAYED_SUMMARY"

    monkeypatch.setattr(agent, "_summarize_for_history", _slow_summary)

    providers([[
        make_response(content="", tool_calls=[{"id": "t", "name": "big_tool"}]),
        make_response(content="done"),
    ]])

    async def _scenario():
        # run() must return BEFORE the summary task completes.
        response, _, turn, pending = await agent.run("go", [])
        assert response == "done"
        assert started.is_set(), "summary task was never scheduled"
        assert len(pending) == 1, f"expected 1 pending summary, got {pending}"
        msg_dict, task = pending[0]
        assert not task.done(), "summary task blocked the turn return"

        # Raw content is still stored pre-finalization.
        tool_msg = [m for m in turn if m.get("role") == "tool"][0]
        assert tool_msg is msg_dict, "pending entry must reference the stored tool message"
        assert tool_msg["content"] == raw, (
            "turn returned with raw content — expected; summary fires in background"
        )

        # Release the summary, drain it, apply (mirrors what main.py does).
        release.set()
        summary = await task
        msg_dict["content"] = summary

        # In-memory patch is visible through the same reference.
        assert tool_msg["content"].startswith("[history summary of big_tool]")
        assert "DELAYED_SUMMARY" in tool_msg["content"]

    asyncio.run(_scenario())


def test_short_tool_output_is_not_summarized(monkeypatch, tmp_system_prompt, providers):
    """Tool outputs below the threshold bypass summarization entirely."""
    _install_fake_tool(monkeypatch, "small_tool", lambda: "short result")

    async def _should_not_run(*a, **kw):
        raise AssertionError("summarizer must not be called for short output")
    monkeypatch.setattr(agent, "_summarize_for_history", _should_not_run)

    providers([[
        make_response(content="", tool_calls=[{"id": "t", "name": "small_tool"}]),
        make_response(content="ok"),
    ]])
    _, _, turn, _ = asyncio.run(agent.run("go", []))
    tool_msg = [m for m in turn if m.get("role") == "tool"][0]
    assert tool_msg["content"] == "short result"


# ── 3. Repair call on empty visible ──────────────────────────────────────────

def test_repair_call_on_empty_visible(monkeypatch, tmp_system_prompt, providers):
    """When model output is only thinking-tags, a non-tool repair call fires."""
    p = providers([[
        make_response(content="<thought>scratch</thought>"),  # visible == ""
        make_response(content="the real answer"),             # repair response
    ]])

    response, _, _, _ = asyncio.run(agent.run("go", []))
    assert response == "the real answer"

    # Second call must have been made with tools disabled (that's the repair
    # contract — DESIGN §4.3).
    assert p.calls[1]["tools"] is None, (
        "repair call must disable tools to prevent recursion"
    )
    # And the repair prompt must have been appended as a user message
    last_user = [m for m in p.calls[1]["messages"] if m["role"] == "user"][-1]
    assert last_user["content"] == agent._REPAIR_USER


def test_repair_not_triggered_when_visible_text_present(monkeypatch, tmp_system_prompt, providers):
    p = providers([[make_response(content="<thought>scratch</thought>hi there")]])
    response, _, _, _ = asyncio.run(agent.run("go", []))
    assert response == "hi there"
    assert len(p.calls) == 1, "no repair should fire when visible text is present"


def test_non_streaming_thinking_content_stripped_from_turn_messages(
    monkeypatch, tmp_system_prompt, providers
):
    """DESIGN §6.3: the final assistant entry in turn_messages must hold only
    visible text — thinking blocks must be stripped before persistence.
    """
    providers([[make_response(content="<thinking>secret</thinking>hello")]])
    _, _, turn, _ = asyncio.run(agent.run("go", []))

    final_assistant = [m for m in turn if m["role"] == "assistant"][-1]
    assert final_assistant["content"] == "hello", (
        f"turn_messages stored raw thinking content: {final_assistant['content']!r}"
    )
    assert "<thinking>" not in (final_assistant["content"] or ""), (
        "thinking tags must be stripped from stored assistant content (§6.3)"
    )


# ── 4. Multi-iteration tool loop ─────────────────────────────────────────────

def test_multi_iteration_tool_loop(monkeypatch, tmp_system_prompt, providers):
    """Two tool calls across three iterations, final answer propagates."""
    calls_seen = []

    def _tool():
        calls_seen.append(None)
        return f"result {len(calls_seen)}"

    _install_fake_tool(monkeypatch, "t", _tool)
    providers([[
        make_response(content="", tool_calls=[{"id": "a", "name": "t"}]),
        make_response(content="", tool_calls=[{"id": "b", "name": "t"}]),
        make_response(content="done"),
    ]])

    response, _, turn, _ = asyncio.run(agent.run("go", []))
    assert response == "done"
    assert len(calls_seen) == 2
    tool_msgs = [m for m in turn if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == ["result 1", "result 2"]


# ── 5. Max-iterations termination ────────────────────────────────────────────

def test_max_tool_iterations_terminates(monkeypatch, tmp_system_prompt, providers):
    """A runaway tool-calling model must not hang the loop indefinitely."""
    _install_fake_tool(monkeypatch, "t", lambda: "x")

    # Build a script that always returns a tool call — exceeds MAX_TOOL_ITERATIONS
    script = [
        make_response(content="", tool_calls=[{"id": f"tc_{i}", "name": "t"}])
        for i in range(agent.MAX_TOOL_ITERATIONS + 2)
    ]
    providers([script])

    response, _, _, _ = asyncio.run(agent.run("go", []))
    assert "max tool iterations" in response.lower()


# ── 6. Provider fallback ─────────────────────────────────────────────────────

def test_provider_fallback_on_retryable_error(monkeypatch, tmp_system_prompt, providers):
    """Provider A raises a retryable error 3× → loop moves to provider B."""
    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    from openai import APIConnectionError
    import httpx

    err = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    comps = providers([
        [err, err, err],                       # provider A exhausts all 3 retries
        [make_response("ok from B")],          # provider B succeeds
    ], names=["a", "b"])

    response, provider, _, _ = asyncio.run(agent.run("hi", []))
    assert provider == "b"
    assert "ok from B" in response


def test_all_providers_exhausted_raises_cleanly(monkeypatch, tmp_system_prompt, providers):
    """When every provider in the chain fails retryably, run() surfaces a
    RuntimeError. main.py converts this to HTTP 500 for the client.
    """
    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    from openai import APIConnectionError
    import httpx

    err = APIConnectionError(request=httpx.Request("POST", "https://x"))
    providers([[err, err, err], [err, err, err], [err, err, err]],
              names=["a", "b", "c"])

    with pytest.raises(RuntimeError, match="providers exhausted"):
        asyncio.run(agent.run("hi", []))


def test_tool_exception_returned_as_error_string(monkeypatch, tmp_system_prompt, providers):
    """A tool that raises must not crash the loop — the error becomes the tool
    message content so the model can react or report it. §5 contract.
    """
    def _explode(**kwargs):
        raise ValueError("database offline")
    _install_fake_tool(monkeypatch, "flaky", _explode)

    p = providers([[
        make_response(content="", tool_calls=[{"id": "t", "name": "flaky"}]),
        make_response(content="tool broke, reporting back"),
    ]])

    response, _, turn, _ = asyncio.run(agent.run("go", []))
    assert response == "tool broke, reporting back"

    # Iteration 2 must have seen the error string in the tool message
    call2_msgs = p.calls[1]["messages"]
    tool_msg = [m for m in call2_msgs if m.get("role") == "tool"][0]
    assert "Error in flaky" in tool_msg["content"]
    assert "database offline" in tool_msg["content"]


def test_unknown_tool_name_returns_error_string(monkeypatch, tmp_system_prompt, providers):
    """Model hallucinating a nonexistent tool name is a recoverable event."""
    monkeypatch.setattr(agent, "TOOL_FUNCTIONS", {})  # no tools registered

    providers([[
        make_response(content="", tool_calls=[{"id": "t", "name": "made_up_tool"}]),
        make_response(content="done"),
    ]])

    response, _, turn, _ = asyncio.run(agent.run("go", []))
    assert response == "done"
    tool_msg = [m for m in turn if m.get("role") == "tool"][0]
    assert "Unknown tool" in tool_msg["content"]
    assert "made_up_tool" in tool_msg["content"]


def test_api_error_skips_provider_immediately(monkeypatch, tmp_system_prompt, providers):
    """A non-retryable APIError should skip to the next provider without retrying."""
    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    from openai import APIError
    import httpx

    err = APIError("boom", request=httpx.Request("POST", "https://x"), body=None)

    comp_a, comp_b = providers([
        [err],                            # one APIError is enough to skip
        [make_response("ok from B")],
    ], names=["a", "b"])

    _, provider, _, _ = asyncio.run(agent.run("hi", []))
    assert provider == "b"
    # Provider A should have been called exactly once (no retries on APIError)
    assert len(comp_a.calls) == 1, f"expected 1 call on A, got {len(comp_a.calls)}"


# ── 7. Streaming ─────────────────────────────────────────────────────────────

async def _collect_stream(user_msg, history=None):
    events = []
    async for ev in agent.run_stream(user_msg, history or []):
        events.append(ev)
    return events


def test_streaming_emits_text_chunks_and_done(monkeypatch, tmp_system_prompt, providers):
    providers([[[text_chunk("hello "), text_chunk("world")]]])
    events = asyncio.run(_collect_stream("hi"))
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "text_chunk")
    assert text == "hello world"


def test_streaming_strips_thinking_tags_from_visible_text(monkeypatch, tmp_system_prompt, providers):
    chunks = [text_chunk("<thought>scratch</thought>"), text_chunk("visible")]
    providers([[chunks]])
    events = asyncio.run(_collect_stream("hi"))
    text = "".join(e["text"] for e in events if e["type"] == "text_chunk")
    assert "scratch" not in text
    assert text.endswith("visible"), text


def test_streaming_tool_call_event_fires_and_result_feeds_next_iter(
    monkeypatch, tmp_system_prompt, providers
):
    """Streaming: tool_call + tool_result events appear and result flows into
    iteration 2's messages."""
    _install_fake_tool(monkeypatch, "fake_tool", lambda: "STREAMED_RESULT")

    iter1 = [tool_chunk(index=0, id="tc_1", name="fake_tool", arguments="{}")]
    iter2 = [text_chunk("after tool")]

    p = providers([[iter1, iter2]])

    events = asyncio.run(_collect_stream("go"))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    # Iteration 2's messages should include the tool result
    call2_msgs = p.calls[1]["messages"]
    tool_msgs = [m for m in call2_msgs if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "STREAMED_RESULT"

    # done event ends the stream
    assert types[-1] == "done"
    done = events[-1]
    roles = [m["role"] for m in done["turn_messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]


# ── 8. Repair path (DESIGN §4.3) ────────────────────────────────────────────

def test_non_streaming_repair_scaffold_not_in_turn_messages(
    monkeypatch, tmp_system_prompt, providers
):
    """Non-streaming repair must mirror streaming: scaffold stays out of history."""
    providers([[
        make_response("<thinking>hidden</thinking>"),
        make_response("real answer"),
    ]])

    final, _, turn, _ = asyncio.run(agent.run("go", []))

    assert final == "real answer"
    user_contents = [m.get("content", "") for m in turn if m["role"] == "user"]
    assert agent._REPAIR_USER not in user_contents, (
        f"repair scaffold leaked into history: {turn}"
    )
    assistants = [m for m in turn if m["role"] == "assistant"]
    assert len(assistants) == 1, f"expected one assistant message, got {assistants}"
    assert (assistants[0].get("content") or "").strip() == "real answer"


def test_streaming_repair_triggers_on_thinking_only(monkeypatch, tmp_system_prompt, providers):
    """A stream consisting entirely of <thinking>...</thinking> produces zero
    visible content. The repair pass must fire, its output replaces the
    empty-visible assistant, and the scaffold (`_REPAIR_USER` user message +
    the empty-visible assistant placeholder) must NOT land in history.
    """
    iter1 = [text_chunk("<thinking>hidden</thinking>")]
    providers([[
        iter1,
        make_response("real answer"),     # repair call (non-streaming)
    ]])
    events = asyncio.run(_collect_stream("go"))
    done = next(e for e in events if e["type"] == "done")

    # (a) final assistant carries the repaired visible answer.
    final_assistant = [m for m in done["turn_messages"] if m["role"] == "assistant"][-1]
    assert (final_assistant.get("content") or "").strip() == "real answer", (
        f"final assistant is not the repaired text: {final_assistant}"
    )

    # (b) the repair scaffold must not land in history.
    user_contents = [m.get("content", "") for m in done["turn_messages"] if m["role"] == "user"]
    assert agent._REPAIR_USER not in user_contents, (
        f"repair scaffold leaked into history: {done['turn_messages']}"
    )
    # And: no assistant message with empty/thinking-only content should remain.
    assistants = [m for m in done["turn_messages"] if m["role"] == "assistant"]
    assert len(assistants) == 1, (
        f"expected exactly one assistant in turn_messages, got {len(assistants)}: "
        f"{assistants}"
    )



def test_streaming_tool_call_name_not_duplicated_across_chunks(
    monkeypatch, tmp_system_prompt, providers
):
    """Gemini/OpenAI stream deltas may repeat the same function name across chunks.
    We should not concatenate duplicates into names like web_searchweb_search.
    """
    _install_fake_tool(monkeypatch, "web_search", lambda query=None, max_results=5: "ok")

    iter1 = [
        tool_chunk(index=0, id="tc_1", name="web_search", arguments=""),
        tool_chunk(index=0, id="tc_1", name="web_search", arguments='{"query":"x"}'),
    ]
    iter2 = [text_chunk("done")]

    p = providers([[iter1, iter2]])

    events = asyncio.run(_collect_stream("go"))
    assert events[-1]["type"] == "done"

    # Second provider call receives the tool message from round 1.
    call2_msgs = p.calls[1]["messages"]
    tool_msgs = [m for m in call2_msgs if m.get("role") == "tool"]
    assert tool_msgs, f"expected tool message in 2nd call, got {call2_msgs}"
    assert tool_msgs[0]["name"] == "web_search"
    assert "Unknown tool" not in tool_msgs[0]["content"]


def test_streaming_tool_call_arguments_accumulate_across_chunks(
    monkeypatch, tmp_system_prompt, providers
):
    """Arguments may stream in fragments; they must be concatenated in order.

    Regression: a prior suffix-only dedup path accidentally dropped argument
    fragments, producing `{}` and triggering extra malformed tool rounds.
    """
    seen_args = {}

    def _fake_web_search(**kwargs):
        seen_args.update(kwargs)
        return "ok"

    _install_fake_tool(monkeypatch, "web_search", _fake_web_search)

    iter1 = [
        tool_chunk(index=0, id="tc_1", name="web_search", arguments='{"query":"'),
        tool_chunk(index=0, id="tc_1", name="web_search", arguments='x","max_results":3}'),
    ]
    iter2 = [text_chunk("done")]

    providers([[iter1, iter2]])
    events = asyncio.run(_collect_stream("go"))
    assert events[-1]["type"] == "done"
    assert seen_args == {"query": "x", "max_results": 3}


