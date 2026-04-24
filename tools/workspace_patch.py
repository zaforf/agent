"""Exact workspace file edit tool (`workspace_search_replace`).

This is the primary code-edit primitive. It performs exact substring replacement
against files under the same root as `shell_exec` (`tools.shell.WORKSPACE`).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tools.shell import WORKSPACE


def _workspace_target(rel: str) -> Path:
    if not rel:
        raise ValueError("invalid path")
    if rel.startswith("/"):
        raise ValueError("absolute paths are not allowed")
    target = (WORKSPACE / rel).resolve()
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

    rel = path.strip().replace("\\", "/").lstrip("./")
    target = _workspace_target(rel)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {target.relative_to(WORKSPACE.resolve()).as_posix()}")

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
                "Primary code-edit tool: exact string replace in a workspace file. "
                "Copy old_string verbatim from shell_exec output (indentation/newlines matter). "
                "Prefer multiple thoughtful sequential calls over risky bulk edits. "
                "If a call fails, adjust snippet/approach instead of repeating identical failing calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace"},
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
