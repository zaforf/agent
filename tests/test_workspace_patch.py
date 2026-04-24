"""Tests for workspace edit tools (search-replace + unified diff)."""
from __future__ import annotations

import pytest

from tools import workspace_patch as wp


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setattr(wp, "WORKSPACE", root)
    return root


def test_apply_creates_file(ws):
    diff = """\
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+alpha
+beta
"""
    assert wp.apply_unified_patch(diff) == "created new.txt"
    assert (ws / "new.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_apply_updates_file_single_hunk(ws):
    (ws / "f.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    diff = """\
--- a/f.py
+++ b/f.py
@@ -1,2 +1,2 @@
-a = 1
+a = 99
 b = 2
"""
    assert wp.apply_unified_patch(diff) == "updated f.py"
    assert (ws / "f.py").read_text(encoding="utf-8") == "a = 99\nb = 2\n"


def test_apply_two_hunks_composes_line_delta(ws):
    """Later hunks use original line numbers; earlier hunks may change file length."""
    (ws / "t.txt").write_text("L1\nL2\nL3\nL4\n", encoding="utf-8")
    diff = """\
--- a/t.txt
+++ b/t.txt
@@ -1,2 +1,2 @@
-L1
+L1y
 L2
@@ -3,2 +3,2 @@
-L3
+L3x
 L4
"""
    wp.apply_unified_patch(diff)
    assert (ws / "t.txt").read_text(encoding="utf-8") == "L1y\nL2\nL3x\nL4\n"


def test_apply_two_hunks_first_inserts_lines(ws):
    """First hunk inserts lines; second hunk's @@ line still refers to pre-patch file."""
    (ws / "t.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    diff = """\
--- a/t.txt
+++ b/t.txt
@@ -2,1 +2,2 @@
 b
+b_extra
@@ -4,1 +5,1 @@
-d
+D
"""
    wp.apply_unified_patch(diff)
    assert (ws / "t.txt").read_text(encoding="utf-8") == "a\nb\nb_extra\nc\nD\n"


def test_apply_deletes_file(ws):
    (ws / "gone.txt").write_text("bye\n", encoding="utf-8")
    diff = """\
--- a/gone.txt
+++ /dev/null
@@ -1,1 +0,0 @@
-bye
"""
    assert "deleted" in wp.apply_unified_patch(diff)
    assert not (ws / "gone.txt").exists()


def test_apply_context_mismatch_raises(ws):
    (ws / "x.txt").write_text("foo\n", encoding="utf-8")
    diff = """\
--- a/x.txt
+++ b/x.txt
@@ -1,1 +1,1 @@
-notfoo
+bar
"""
    with pytest.raises(ValueError, match="context mismatch"):
        wp.apply_unified_patch(diff)


def test_apply_rejects_path_escape(ws):
    diff = """\
--- a/../../../etc/passwd
+++ b/../../../etc/passwd
@@ -1,1 +1,1 @@
 x
+y
"""
    with pytest.raises(ValueError, match="escapes workspace"):
        wp.apply_unified_patch(diff)


def test_apply_rejects_rename(ws):
    (ws / "a.txt").write_text("x\n", encoding="utf-8")
    diff = """\
--- a/a.txt
+++ b/b.txt
@@ -1,1 +1,1 @@
 x
+y
"""
    with pytest.raises(ValueError, match="rename not supported"):
        wp.apply_unified_patch(diff)


def test_apply_rejects_absolute_in_diff(ws):
    diff = """\
--- /abs/foo.txt
+++ /abs/foo.txt
@@ -1,1 +1,1 @@
 a
+b
"""
    with pytest.raises(ValueError, match="absolute path"):
        wp.apply_unified_patch(diff)


def test_apply_multi_file(ws):
    diff = """\
--- /dev/null
+++ b/one.txt
@@ -0,0 +1,1 @@
+1
--- /dev/null
+++ b/two.txt
@@ -0,0 +1,1 @@
+2
"""
    out = wp.apply_unified_patch(diff)
    assert "created one.txt" in out and "created two.txt" in out
    assert (ws / "one.txt").read_text() == "1\n"
    assert (ws / "two.txt").read_text() == "2\n"


def test_apply_preserves_missing_trailing_newline(ws):
    (ws / "nl.txt").write_bytes(b"only")
    diff = """\
--- a/nl.txt
+++ b/nl.txt
@@ -1,1 +1,1 @@
-only
+patched
"""
    wp.apply_unified_patch(diff)
    assert (ws / "nl.txt").read_bytes() == b"patched"


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
