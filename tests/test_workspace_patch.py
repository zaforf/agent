"""Tests for workspace edit tool (`workspace_search_replace`)."""
from __future__ import annotations

import pytest

from tools import workspace_patch as wp


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setattr(wp, "WORKSPACE", root)
    return root


def test_workspace_search_replace_single(ws):
    (ws / "x.py").write_text("foo = 1\n  bar = 2\n", encoding="utf-8")
    out = wp.workspace_search_replace("x.py", "  bar = 2\n", "  bar = 99\n")
    assert "updated" in out
    assert (ws / "x.py").read_text() == "foo = 1\n  bar = 99\n"


def test_workspace_search_replace_requires_unique(ws):
    (ws / "d.txt").write_text("a\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matches 2 times"):
        wp.workspace_search_replace("d.txt", "a\n", "b\n", replace_all=False)


def test_workspace_search_replace_all(ws):
    (ws / "d.txt").write_text("a\na\n", encoding="utf-8")
    wp.workspace_search_replace("d.txt", "a\n", "x\n", replace_all=True)
    assert (ws / "d.txt").read_text() == "x\nx\n"


def test_workspace_search_replace_not_found(ws):
    (ws / "f.txt").write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="old_string not found"):
        wp.workspace_search_replace("f.txt", "bye\n", "ok\n")
