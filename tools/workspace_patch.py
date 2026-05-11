"""Exact workspace file edit tool (`workspace_search_replace`).

This is the primary code-edit primitive. It performs exact substring replacement
against files resolved relative to the shell's current working directory, exactly
as the shell would find them after `ls`. Paths are still security-checked to
remain within WORKSPACE (`tools.shell.WORKSPACE`).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tools.shell import WORKSPACE, get_shell_cwd


def _workspace_target(rel: str) -> Path:
    if not rel:
        raise ValueError("invalid path")
    # Absolute paths are fine as long as they resolve inside the workspace.
    # The relative_to() check below is the real security boundary.
    cwd = get_shell_cwd()
    target = (Path(rel) if rel.startswith("/") else (cwd / rel)).resolve()
    try:
        target.relative_to(WORKSPACE.resolve())
    except ValueError as e:
        raise ValueError(f"path escapes workspace: {rel!r}") from e
    return target


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def workspace_read(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a workspace file with line numbers (1-indexed, inclusive).

    Omit start_line/end_line to read the whole file.  Use a tight range to
    get the exact text you need before a workspace_search_replace call.
    """
    rel = path.strip().replace("\\", "/")
    target = _workspace_target(rel)
    if not target.is_file():
        cwd = get_shell_cwd()
        raise FileNotFoundError(
            f"not a file: {rel!r} (resolved to {target} from cwd={cwd})"
        )
    raw = target.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines(keepends=True)
    total = len(lines)
    s = max(1, start_line) - 1          # 0-indexed start
    e = min(total, end_line) if end_line is not None else total
    selected = lines[s:e]
    if not selected:
        return f"(no lines in range {start_line}-{end_line or total} of {total}-line file)"
    return "".join(f"{s + i + 1}\t{line}" for i, line in enumerate(selected))


def workspace_search_replace(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace `old_string` with `new_string` in a workspace file (exact match).

    Use verbatim text copied from shell output (indentation/newlines must match).
    """
    if not old_string:
        raise ValueError("old_string must be non-empty")

    rel = path.strip().replace("\\", "/")
    target = _workspace_target(rel)
    if not target.is_file():
        cwd = get_shell_cwd()
        raise FileNotFoundError(
            f"not a file: {rel!r} (resolved to {target} from cwd={cwd})"
        )

    content = target.read_text(encoding="utf-8", errors="replace")
    n = content.count(old_string)
    if n == 0:
        raise ValueError(
            f"old_string not found in {rel!r} (copy exact text from the file, including whitespace)"
        )
    if n > 1 and not replace_all:
        raise ValueError(
            f"old_string matches {n} times in {rel!r}; use a longer unique snippet or replace_all=true"
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
        out = f"updated {target.relative_to(WORKSPACE.resolve()).as_posix()} ({n} replacement(s))"
    else:
        new_content = content.replace(old_string, new_string, 1)
        out = f"updated {target.relative_to(WORKSPACE.resolve()).as_posix()}"

    _atomic_write_text(target, new_content)
    return out


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "workspace_read",
            "description": (
                "Read a workspace file with line numbers. "
                "Call this immediately before workspace_search_replace to get the exact text — "
                "never construct old_string from memory. "
                "Use start_line/end_line to narrow to the region you intend to edit; "
                "omit both to read the whole file. "
                "Paths are relative to the shell's current working directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to shell's current working directory"},
                    "start_line": {"type": "integer", "description": "First line to return (1-indexed, default 1)"},
                    "end_line": {"type": "integer", "description": "Last line to return inclusive (default: end of file)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_search_replace",
            "description": (
                "Edit an existing workspace file by exact string replacement. "
                "ALWAYS call workspace_read immediately before this tool to get the verbatim text — "
                "never construct old_string from memory or prior context. "
                "On failure: call workspace_read again, find the exact mismatch, and retry. "
                "Never fall back to shell_exec+heredoc for existing files, even after repeated failures. "
                "shell_exec+heredoc is only for creating a file that does not yet exist. "
                "Paths are relative to the shell's current working directory; "
                "absolute paths within the workspace are also accepted. "
                "Prefer multiple targeted calls over bulk whole-file rewrites."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to shell's current working directory"},
                    "old_string": {"type": "string", "description": "Exact substring to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence (default false)",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    }
]

FUNCTIONS = {
    "workspace_read": workspace_read,
    "workspace_search_replace": workspace_search_replace,
}
