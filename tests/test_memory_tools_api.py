"""Mem0 API compatibility: search/get_all use filters= not top-level user_id."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import tools.memory as memory


def test_recall_uses_filters_not_user_id_kwarg():
    mock_mem = MagicMock()
    mock_mem.search.return_value = {"results": [{"memory": "hello"}]}

    with patch.object(memory, "_get_memory", return_value=mock_mem):
        out = memory.recall("test query")

    assert "hello" in out
    mock_mem.search.assert_called_once()
    call_kw = mock_mem.search.call_args
    assert call_kw[0][0] == "test query"
    assert call_kw[1]["filters"] == {"user_id": memory.USER_ID}
    assert call_kw[1]["top_k"] == 5
    assert "user_id" not in call_kw[1]


def test_list_memories_uses_filters_not_user_id_kwarg():
    mock_mem = MagicMock()
    mock_mem.get_all.return_value = {"results": []}

    with patch.object(memory, "_get_memory", return_value=mock_mem):
        memory.list_memories()

    mock_mem.get_all.assert_called_once_with(filters={"user_id": memory.USER_ID})


def test_get_all_api_uses_filters():
    mock_mem = MagicMock()
    mock_mem.get_all.return_value = {"results": []}

    with patch.object(memory, "_get_memory", return_value=mock_mem):
        memory.get_all()

    mock_mem.get_all.assert_called_once_with(filters={"user_id": memory.USER_ID})
