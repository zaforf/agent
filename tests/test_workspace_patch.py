"""Tests for workspace tools (`workspace_read`, `workspace_search_replace`)."""
from __future__ import annotations

import pytest

import tools.shell as sh
from tools import workspace_patch as wp


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Isolated workspace: tmp root, shell cwd reset to root."""
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setattr(wp, "WORKSPACE", root)
    monkeypatch.setattr(sh, "WORKSPACE", root)
    monkeypatch.setattr(sh, "_shell_cwd", root)
    return root


# ── Basic edits ───────────────────────────────────────────────────────────────

def test_single_replacement(ws):
    (ws / "x.py").write_text("foo = 1\n  bar = 2\n", encoding="utf-8")
    out = wp.workspace_search_replace("x.py", "  bar = 2\n", "  bar = 99\n")
    assert "updated" in out
    assert (ws / "x.py").read_text() == "foo = 1\n  bar = 99\n"


def test_requires_unique_match(ws):
    (ws / "d.txt").write_text("a\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matches 2 times"):
        wp.workspace_search_replace("d.txt", "a\n", "b\n", replace_all=False)


def test_replace_all(ws):
    (ws / "d.txt").write_text("a\na\n", encoding="utf-8")
    wp.workspace_search_replace("d.txt", "a\n", "x\n", replace_all=True)
    assert (ws / "d.txt").read_text() == "x\nx\n"


def test_not_found(ws):
    (ws / "f.txt").write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="old_string not found"):
        wp.workspace_search_replace("f.txt", "bye\n", "ok\n")


# ── CWD-relative path resolution (the primary use case) ──────────────────────

def test_cwd_relative_subdir(ws, monkeypatch):
    """File in subdir is found by bare name when shell cwd is that subdir."""
    sub = ws / "myproject"
    sub.mkdir()
    (sub / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(sh, "_shell_cwd", sub)

    wp.workspace_search_replace("main.py", "x = 1\n", "x = 42\n")
    assert (sub / "main.py").read_text() == "x = 42\n"


def test_cwd_subpath_still_works_from_root(ws):
    """Explicit subdir/file path works when cwd is workspace root."""
    sub = ws / "src"
    sub.mkdir()
    (sub / "app.py").write_text("v = 0\n", encoding="utf-8")

    wp.workspace_search_replace("src/app.py", "v = 0\n", "v = 1\n")
    assert (sub / "app.py").read_text() == "v = 1\n"


def test_dotslash_prefix_works(ws):
    """./file.txt is accepted and resolves correctly."""
    (ws / "f.txt").write_text("old\n", encoding="utf-8")
    wp.workspace_search_replace("./f.txt", "old\n", "new\n")
    assert (ws / "f.txt").read_text() == "new\n"


def test_hidden_file_not_mangled(ws):
    """Dotfiles like .env are no longer stripped by lstrip."""
    (ws / ".env").write_text("SECRET=old\n", encoding="utf-8")
    wp.workspace_search_replace(".env", "SECRET=old\n", "SECRET=new\n")
    assert (ws / ".env").read_text() == "SECRET=new\n"


def test_parent_traversal_rejected(ws):
    """../escape should raise, not silently resolve to workspace root."""
    (ws / "legit.txt").write_text("ok\n", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        wp.workspace_search_replace("../escape.txt", "x", "y")


def test_error_message_includes_resolved_path(ws, monkeypatch):
    """FileNotFoundError should tell the agent where it actually looked."""
    sub = ws / "proj"
    sub.mkdir()
    monkeypatch.setattr(sh, "_shell_cwd", sub)
    with pytest.raises(FileNotFoundError, match="main.py"):
        wp.workspace_search_replace("main.py", "x", "y")


# ── Safety ────────────────────────────────────────────────────────────────────

def test_absolute_path_rejected(ws):
    with pytest.raises(ValueError, match="absolute paths are not allowed"):
        wp.workspace_search_replace("/etc/passwd", "root", "evil")


def test_empty_old_string_rejected(ws):
    (ws / "f.txt").write_text("content\n", encoding="utf-8")
    with pytest.raises(ValueError):
        wp.workspace_search_replace("f.txt", "", "new")


def test_atomic_write_preserves_on_error(ws):
    """A failed replacement leaves the original file untouched."""
    original = "hello\n"
    (ws / "safe.txt").write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        wp.workspace_search_replace("safe.txt", "nothere", "x")
    assert (ws / "safe.txt").read_text() == original


# ── workspace_read ────────────────────────────────────────────────────────────

def test_read_whole_file(ws):
    (ws / "r.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    out = wp.workspace_read("r.txt")
    assert "1\talpha\n" in out
    assert "2\tbeta\n" in out
    assert "3\tgamma\n" in out


def test_read_line_range(ws):
    (ws / "r.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    out = wp.workspace_read("r.txt", start_line=2, end_line=3)
    assert "2\tb\n" in out
    assert "3\tc\n" in out
    assert "1\ta" not in out
    assert "4\td" not in out


def test_read_missing_file_raises(ws):
    with pytest.raises(FileNotFoundError):
        wp.workspace_read("nope.txt")


def test_read_cwd_relative(ws, monkeypatch):
    sub = ws / "sub"
    sub.mkdir()
    (sub / "f.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(sh, "_shell_cwd", sub)
    out = wp.workspace_read("f.py")
    assert "x = 1" in out
