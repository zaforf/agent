"""Unit tests for ambient memory prefetch (_prefetch_memories, _memories_system_block)."""
from __future__ import annotations

import asyncio
import pytest

import agent


# ── _memories_system_block ────────────────────────────────────────────────────

def test_memories_block_format():
    block = agent._memories_user_block(["Zafir prefers concise answers", "Uses LaTeX for math"])
    assert block["role"] == "user"
    assert "• Zafir prefers concise answers" in block["content"]
    assert "• Uses LaTeX for math" in block["content"]
    assert "memories" in block["content"].lower()


def test_memories_block_single():
    block = agent._memories_user_block(["one fact"])
    assert block["content"].count("•") == 1


# ── _prefetch_memories ────────────────────────────────────────────────────────

def test_prefetch_returns_filtered_results(monkeypatch):
    """Only memories above the threshold are returned."""
    def fake_recall_prefetch(query, *, top_k, threshold):
        all_results = ["high-score memory", "low-score memory"]
        # Simulate threshold filtering already done inside recall_prefetch
        return ["high-score memory"]

    monkeypatch.setattr("tools.memory.recall_prefetch", fake_recall_prefetch)
    results = asyncio.run(agent._prefetch_memories("what is 2+2", []))
    assert results == ["high-score memory"]


def test_prefetch_enriches_query_with_last_assistant_turn(monkeypatch):
    """Query sent to recall_prefetch includes the last assistant message snippet."""
    captured = {}

    def fake_recall_prefetch(query, *, top_k, threshold):
        captured["query"] = query
        return []

    monkeypatch.setattr("tools.memory.recall_prefetch", fake_recall_prefetch)
    history = [
        {"role": "user", "content": "what is LaTeX?"},
        {"role": "assistant", "content": "LaTeX is a typesetting system for math."},
    ]
    asyncio.run(agent._prefetch_memories("help with homework", history))
    assert "help with homework" in captured["query"]
    assert "LaTeX" in captured["query"]  # cross-domain bridge from last assistant turn


def test_prefetch_skips_assistant_snippet_when_no_history(monkeypatch):
    captured = {}

    def fake_recall_prefetch(query, *, top_k, threshold):
        captured["query"] = query
        return []

    monkeypatch.setattr("tools.memory.recall_prefetch", fake_recall_prefetch)
    asyncio.run(agent._prefetch_memories("hello world", []))
    assert captured["query"].strip() == "hello world"


def test_prefetch_silent_on_exception(monkeypatch):
    """A failing memory store must not raise — returns empty list."""
    def boom(query, *, top_k, threshold):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr("tools.memory.recall_prefetch", boom)
    results = asyncio.run(agent._prefetch_memories("anything", []))
    assert results == []


def test_prefetch_empty_when_no_memories(monkeypatch):
    monkeypatch.setattr("tools.memory.recall_prefetch", lambda *a, **kw: [])
    results = asyncio.run(agent._prefetch_memories("anything", []))
    assert results == []
