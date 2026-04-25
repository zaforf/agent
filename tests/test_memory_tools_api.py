"""mem0ai 2.x API: search/get_all require filters= and top_k on search (not user_id=/limit=)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import tools.memory as memory


def test_recall_calls_search_with_filters_and_top_k():
    mock_mem = MagicMock()
    mock_mem.search.return_value = {"results": [{"memory": "x"}]}

    with patch.object(memory, "_get_memory", return_value=mock_mem):
        memory.recall("q")

    mock_mem.search.assert_called_once_with(
        "q",
        filters={"user_id": memory.USER_ID},
        top_k=5,
    )


def test_list_memories_calls_get_all_with_filters():
    mock_mem = MagicMock()
    mock_mem.get_all.return_value = {"results": []}

    with patch.object(memory, "_get_memory", return_value=mock_mem):
        memory.list_memories()

    mock_mem.get_all.assert_called_once_with(filters={"user_id": memory.USER_ID})


def test_get_all_api_calls_get_all_with_filters():
    mock_mem = MagicMock()
    mock_mem.get_all.return_value = {"results": []}

    with patch.object(memory, "_get_memory", return_value=mock_mem):
        memory.get_all()

    mock_mem.get_all.assert_called_once_with(filters={"user_id": memory.USER_ID})


def test_remember_persists_with_infer_flag_from_config(monkeypatch):
    """remember() passes config.MEM0_REMEMBER_INFER to mem0.add (default False)."""
    mock_mem = MagicMock()
    mock_mem.add.return_value = {"results": [{"id": "1", "memory": "x", "event": "ADD"}]}

    monkeypatch.setattr(memory.config, "MEM0_REMEMBER_INFER", False)
    with patch.object(memory, "_get_memory", return_value=mock_mem):
        memory.remember("favorite color is blue", category="preference")
    mock_mem.add.assert_called_once_with(
        "favorite color is blue",
        user_id=memory.USER_ID,
        metadata={"category": "preference"},
        infer=False,
    )

    mock_mem.reset_mock()
    monkeypatch.setattr(memory.config, "MEM0_REMEMBER_INFER", True)
    with patch.object(memory, "_get_memory", return_value=mock_mem):
        memory.remember("a", category="fact")
    mock_mem.add.assert_called_once_with(
        "a", user_id=memory.USER_ID, metadata={"category": "fact"}, infer=True
    )
