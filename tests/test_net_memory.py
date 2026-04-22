"""Live Mem0 + Qdrant + Gemini embedding path (no local sentence-transformers).

Skipped in default pytest (``@pytest.mark.net``). Requires:

- ``GEMINI_API_KEY`` — Gemini Embedding API
- Qdrant at ``QDRANT_HOST`` / ``QDRANT_PORT`` (e.g. local or docker)

Uses ``infer=False`` so Mem0 skips Groq LLM extraction and only exercises
embedding + vector store (minimal surface for "embeddings work").
"""
from __future__ import annotations

import socket
import uuid

import pytest

import config
from mem0 import Memory

from tools import memory as memory_tools
from tools.memory import USER_ID, mem0_config_dict

pytestmark = pytest.mark.net


def _qdrant_reachable() -> bool:
    try:
        socket.create_connection((config.QDRANT_HOST, config.QDRANT_PORT), timeout=1.0).close()
    except OSError:
        return False
    return True


@pytest.mark.skipif(not config.GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
@pytest.mark.skipif(not _qdrant_reachable(), reason="Qdrant not reachable at QDRANT_HOST:QDRANT_PORT")
def test_mem0_gemini_embedding_add_without_infer_llm():
    """Round-trip: embed via Gemini API, write to Qdrant, search returns the text."""
    probe = f"pytest embedding probe {uuid.uuid4().hex[:12]}"
    mem = Memory.from_config(mem0_config_dict())
    mem.add(probe, user_id=USER_ID, metadata={"category": "fact"}, infer=False)
    out = mem.search(probe, filters={"user_id": USER_ID}, top_k=3)
    results = out.get("results") or []
    texts = [r.get("memory", "") for r in results]
    assert any(probe in t for t in texts), f"probe not in search results: {texts}"


@pytest.mark.skipif(not config.GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
@pytest.mark.skipif(not _qdrant_reachable(), reason="Qdrant not reachable at QDRANT_HOST:QDRANT_PORT")
def test_remember_tool_then_list_and_recall_sees_content(monkeypatch):
    """End-to-end: tools.remember uses infer=False so vectors exist; list/recall find them."""
    monkeypatch.setattr(memory_tools, "_memory", None)
    probe = f"MEM_ROUNDTRIP_{uuid.uuid4().hex}"
    memory_tools.remember(probe, category="fact")
    listed = memory_tools.list_memories()
    assert probe in listed, f"list_memories missing probe: {listed[:500]!r}"
    recalled = memory_tools.recall(probe)
    assert probe in recalled or probe[:32] in recalled, f"recall missing probe: {recalled!r}"
