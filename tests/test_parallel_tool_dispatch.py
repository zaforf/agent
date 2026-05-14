"""Parallel read-only tool batches (agent._run_tool_specs_to_results)."""

from __future__ import annotations

import asyncio
import time

import pytest

import agent
import config
from tests.conftest import make_response, text_chunk, tool_chunk


def test_tool_batch_parallel_eligible_requires_two_safe_tools():
    assert agent._tool_batch_parallel_eligible(["recall", "list_memories"]) is True
    assert agent._tool_batch_parallel_eligible(["recall"]) is False
    assert agent._tool_batch_parallel_eligible(["recall", "workspace_search_replace"]) is False


def test_fake_provider_records_parallel_tool_calls_flag(monkeypatch, tmp_system_prompt, providers):
    monkeypatch.setattr(config, "AGENT_PARALLEL_TOOL_CALLS", True)
    comp = providers([[make_response("x")]])
    asyncio.run(agent.run("hi", []))
    assert comp.calls[0].get("parallel_tool_calls") is True


def test_parallel_disabled_skips_parallel_tool_calls_kwarg(monkeypatch, tmp_system_prompt, providers):
    monkeypatch.setattr(config, "AGENT_PARALLEL_TOOL_CALLS", False)
    comp = providers([[make_response("x")]])
    asyncio.run(agent.run("hi", []))
    assert "parallel_tool_calls" not in comp.calls[0]


@pytest.mark.parametrize(
    "names,expect_parallel",
    [
        (["recall", "list_memories"], True),
        (["recall", "remember"], False),
        (["workspace_read", "workspace_read"], True),
    ],
)
def test_run_tool_specs_respects_parallel_gate(monkeypatch, names, expect_parallel):
    """When eligible, two slow tools overlap; when not, they run one after the other."""

    state = {"active": 0, "max_active": 0}

    async def slow_run(name, args):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.08)
        state["active"] -= 1
        return f"ok-{name}"

    monkeypatch.setattr(agent, "_run_tool_async", slow_run)
    monkeypatch.setattr(config, "AGENT_PARALLEL_TOOL_CALLS", True)

    specs = [("id-a", names[0], {}), ("id-b", names[1], {})]
    t0 = time.perf_counter()
    rows = asyncio.run(agent._run_tool_specs_to_results(specs))
    elapsed = time.perf_counter() - t0

    assert len(rows) == 2
    if expect_parallel:
        assert state["max_active"] == 2, "tools should overlap when parallel-safe"
        assert elapsed < 0.14, f"parallel wall-clock too high: {elapsed:.3f}s"
    else:
        assert state["max_active"] == 1
        assert elapsed >= 0.12, f"sequential wall-clock too low: {elapsed:.3f}s"


async def _collect_stream(user_msg, history=None):
    events = []
    async for ev in agent.run_stream(user_msg, history or []):
        events.append(ev)
    return events


def test_streaming_emits_two_tool_pairs(monkeypatch, tmp_system_prompt, providers):
    """Two read-only tools in one stream batch produce call/result pairs."""
    monkeypatch.setattr(config, "AGENT_PARALLEL_TOOL_CALLS", True)

    async def echo(name, args):
        return f"result-{name}"

    monkeypatch.setattr(agent, "_run_tool_async", echo)

    iter1 = [
        tool_chunk(0, "tc_a", "recall", '{"query":"x"}'),
        tool_chunk(1, "tc_b", "list_memories", "{}"),
    ]
    iter2 = [text_chunk("done")]
    providers([[iter1, iter2]])

    events = asyncio.run(_collect_stream("go"))
    types = [e["type"] for e in events]
    assert types.count("tool_call") == 2
    assert types.count("tool_result") == 2
    done = events[-1]
    tool_msgs = [m for m in done["turn_messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "tc_a"
    assert tool_msgs[1]["tool_call_id"] == "tc_b"


def test_streaming_same_index_parallel_tools_split(monkeypatch, tmp_system_prompt, providers):
    """Gemini-style: every delta uses index=0 but names are distinct tools."""
    monkeypatch.setattr(config, "AGENT_PARALLEL_TOOL_CALLS", True)

    async def echo(name, args):
        return f"ok-{name}"

    monkeypatch.setattr(agent, "_run_tool_async", echo)

    iter1 = [
        tool_chunk(0, "", "workspace_read", '{"path":"DESIGN.md"}'),
        tool_chunk(0, "", "workspace_grep", '{"path":".","pattern":"def"}'),
        tool_chunk(0, "", "recall", '{"query":"prefs"}'),
    ]
    iter2 = [text_chunk("done")]
    providers([[iter1, iter2]])

    events = asyncio.run(_collect_stream("go"))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 3
    assert {c["name"] for c in calls} == {"workspace_read", "workspace_grep", "recall"}
    done = events[-1]
    tool_msgs = [m for m in done["turn_messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 3
