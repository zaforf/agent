"""Mem0 memory stack configuration (Gemini API embeddings, no local sentence-transformers)."""
from __future__ import annotations

import agent


def test_mem0_config_uses_gemini_embedder_and_dims(monkeypatch):
    import config
    import tools.memory as memory

    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    monkeypatch.setattr(config, "GEMINI_EMBEDDING_DIMS", 768)
    monkeypatch.setattr(config, "MEM0_QDRANT_COLLECTION", "agent_memories_gemini")
    monkeypatch.setattr(config, "QDRANT_HOST", "localhost")
    monkeypatch.setattr(config, "QDRANT_PORT", 6333)
    monkeypatch.setattr(config, "MEM0_LLM_MODEL", "gemma-4-26b-a4b-it")

    cfg = memory.mem0_config_dict()
    assert cfg["llm"]["provider"] == "gemini"
    assert cfg["llm"]["config"]["model"] == "gemma-4-26b-a4b-it"
    assert cfg["llm"]["config"]["api_key"] == "test-key"
    assert cfg["embedder"]["provider"] == "gemini"
    assert cfg["embedder"]["config"]["model"] == "models/gemini-embedding-001"
    assert cfg["embedder"]["config"]["embedding_dims"] == 768
    assert cfg["embedder"]["config"]["api_key"] == "test-key"
    assert cfg["vector_store"]["config"]["collection_name"] == "agent_memories_gemini"
    assert cfg["vector_store"]["config"]["embedding_model_dims"] == 768


def test_memory_tools_are_blocking_for_async_loop():
    """Mem0 work must not block the event loop (gunicorn worker timeouts)."""
    assert "remember" in agent._BLOCKING_SYNC_TOOLS
    assert "recall" in agent._BLOCKING_SYNC_TOOLS
    assert "list_memories" in agent._BLOCKING_SYNC_TOOLS
    assert "delete_memory" in agent._BLOCKING_SYNC_TOOLS
