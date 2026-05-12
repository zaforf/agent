"""Tests for workspace tools (`workspace_grep`, `workspace_read`, `workspace_search_replace`)."""
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

def test_absolute_path_outside_workspace_rejected(ws):
    # /etc/passwd escapes the workspace — caught by relative_to() check.
    with pytest.raises((ValueError, FileNotFoundError)):
        wp.workspace_search_replace("/etc/passwd", "root", "evil")


def test_absolute_path_inside_workspace_allowed(ws):
    (ws / "a.txt").write_text("old\n", encoding="utf-8")
    wp.workspace_search_replace(str(ws / "a.txt"), "old\n", "new\n")
    assert (ws / "a.txt").read_text() == "new\n"


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
    assert "1 | alpha\n" in out
    assert "2 | beta\n" in out
    assert "3 | gamma\n" in out


def test_read_line_range(ws):
    (ws / "r.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    out = wp.workspace_read("r.txt", start_line=2, end_line=3)
    assert "2 | b\n" in out
    assert "3 | c\n" in out
    assert "1 | a" not in out
    assert "4 | d" not in out


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


# ── workspace_grep ────────────────────────────────────────────────────────────

def test_grep_finds_match(ws):
    (ws / "g.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
    out = wp.workspace_grep("g.py", "def foo")
    assert "def foo" in out
    assert "1 | " in out  # line 1


def test_grep_returns_line_numbers(ws):
    (ws / "g.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    out = wp.workspace_grep("g.py", "c", context_lines=1)
    assert "2 | b" in out   # context before
    assert "3 | c" in out   # match
    assert "4 | d" in out   # context after


def test_grep_no_match(ws):
    (ws / "g.py").write_text("hello world\n", encoding="utf-8")
    out = wp.workspace_grep("g.py", "zzz_no_match")
    assert "no matches" in out


def test_grep_regex_pattern(ws):
    (ws / "g.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    out = wp.workspace_grep("g.py", r"[xyz] = [23]", context_lines=0)
    assert "y = 2" in out
    assert "z = 3" in out
    assert "x = 1" not in out


def test_grep_merges_overlapping_context(ws):
    # Two matches close enough that their context windows overlap — should produce
    # one contiguous block, not two blocks with a -- separator.
    lines = "\n".join(f"line{i}" for i in range(1, 11))
    (ws / "g.py").write_text(lines + "\n", encoding="utf-8")
    out = wp.workspace_grep("g.py", r"line[23]", context_lines=2)
    assert "--" not in out   # merged into one block


def test_grep_invalid_regex_treated_as_literal(ws):
    (ws / "g.py").write_text("price: $5.00\n", encoding="utf-8")
    out = wp.workspace_grep("g.py", "$5.00")  # would be invalid regex if not escaped
    assert "$5.00" in out


def test_grep_missing_file_raises(ws):
    with pytest.raises(FileNotFoundError):
        wp.workspace_grep("nope.py", "anything")
