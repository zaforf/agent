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
    if rel.startswith("/"):
        raise ValueError("absolute paths are not allowed")
    # Resolve relative to the shell's current working directory so the agent
    # can pass bare filenames (e.g. "main.py") after cd-ing into a subdir.
    cwd = get_shell_cwd()
    target = (cwd / rel).resolve()
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
            "name": "workspace_search_replace",
            "description": (
                "Primary code-edit tool — use for ALL edits to existing files. "
                "shell_exec+echo/heredoc is only acceptable when creating a file that does not yet exist; "
                "every subsequent change must use this tool. "
                "Before calling, retrieve the exact lines with shell_exec (grep -n or sed -n) — "
                "never construct old_string from memory. "
                "Paths are relative to the shell's current working directory: "
                "after cd myproject/, pass 'main.py' not 'myproject/main.py'. "
                "Prefer multiple targeted sequential calls over bulk whole-file rewrites. "
                "After a failed call, adjust the snippet or path; never retry identical failing calls."
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

FUNCTIONS = {"workspace_search_replace": workspace_search_replace}
