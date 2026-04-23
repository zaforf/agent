"""Unified-diff apply for files under the shell workspace (`tools.shell.WORKSPACE`).

`apply_unified_patch` applies a **git-style unified diff** (``---`` / ``+++`` / ``@@`` hunks). Pure
Python (no ``patch(1)``). Read files with ``shell_exec`` (e.g. ``cat``, ``head``, ``sed``) before
building a patch so hunks match.

**Design choices**

- **Paths:** ``a/`` and ``b/`` prefixes from ``git diff`` are stripped; the file is resolved under
  ``WORKSPACE``. Absolute paths in ``---`` / ``+++`` are rejected.
- **Atomic writes:** temp file in the target directory, then ``os.replace``.
- **Hunks:** applied **bottom-up** (highest ``old_start`` first) so line indices stay stable.
- **Renames:** ``---`` and ``+++`` must refer to the same workspace-relative path (no rename in one patch).
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tools.shell import WORKSPACE

log = logging.getLogger(__name__)

_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    new_count: int
    raw_body: tuple[str, ...]  # lines like " foo", "-bar", "+baz" (prefix + space + rest)


def _reject_abs_path_marker(line: str, which: str) -> str:
    rest = line[4:].rstrip("\r\n")
    if not rest or rest == "/dev/null":
        return rest
    path_token = rest.split("\t", 1)[0].strip()
    if path_token.startswith("/"):
        raise ValueError(
            f"{which} path must be relative to workspace (got absolute path in diff: {path_token!r})"
        )
    return path_token


def _normalize_diff_path(raw: str) -> str:
    if raw == "/dev/null":
        return raw
    p = raw.replace("\\", "/").lstrip("./")
    for prefix in ("a/", "b/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    return p


def _workspace_target(rel: str) -> Path:
    if not rel or rel == "/dev/null":
        raise ValueError("invalid path in patch")
    if rel.startswith("/"):
        raise ValueError("absolute paths are not allowed in workspace patches")
    target = (WORKSPACE / rel).resolve()
    try:
        target.relative_to(WORKSPACE.resolve())
    except ValueError as e:
        raise ValueError(f"path escapes workspace: {rel!r}") from e
    return target


def _split_hunk_sides(raw_body: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split hunk into old-file lines and new-file lines (content only, no prefixes)."""
    old_lines: list[str] = []
    new_lines: list[str] = []
    for raw in raw_body:
        if raw.startswith("\\"):
            # "\ No newline at end of file" — informational only
            continue
        if len(raw) < 1:
            continue
        kind = raw[0]
        text = raw[1:] if len(raw) > 1 else ""
        if kind == " ":
            old_lines.append(text)
            new_lines.append(text)
        elif kind == "-":
            old_lines.append(text)
        elif kind == "+":
            new_lines.append(text)
        else:
            raise ValueError(f"invalid hunk line (expected ' ', '-', or '+', got {kind!r})")
    return old_lines, new_lines


def _parse_hunks(lines: list[str], start: int) -> tuple[list[_Hunk], int]:
    hunks: list[_Hunk] = []
    i = start
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("--- "):
            break
        m = _HUNK_RE.match(line)
        if not m:
            i += 1
            continue
        old_s = int(m.group(1))
        old_c = int(m.group(2)) if m.group(2) is not None else 1
        new_c = int(m.group(4)) if m.group(4) is not None else 1
        i += 1
        body: list[str] = []
        while i < n:
            bl = lines[i]
            if bl.startswith("--- ") or _HUNK_RE.match(bl):
                break
            body.append(bl.rstrip("\r\n"))
            i += 1
        hunks.append(
            _Hunk(old_start=old_s, old_count=old_c, new_count=new_c, raw_body=tuple(body))
        )
    return hunks, i


def _parse_file_segment(lines: list[str], idx: int) -> tuple[str | None, str | None, list[_Hunk], int]:
    if idx >= len(lines) or not lines[idx].startswith("--- "):
        raise ValueError("malformed unified diff: expected --- line")
    old_raw = _reject_abs_path_marker(lines[idx], "---")
    idx += 1
    if idx >= len(lines) or not lines[idx].startswith("+++ "):
        raise ValueError("malformed unified diff: expected +++ line after ---")
    new_raw = _reject_abs_path_marker(lines[idx], "+++")
    idx += 1
    hunks, idx = _parse_hunks(lines, idx)
    return old_raw, new_raw, hunks, idx


def _apply_one_file(path: Path, hunks: list[_Hunk], is_new: bool) -> None:
    if is_new:
        if path.exists():
            raise ValueError(
                f"patch creates {path.relative_to(WORKSPACE.resolve())} but file already exists"
            )
        new_lines: list[str] = []
        for h in hunks:
            _, new_side = _split_hunk_sides(h.raw_body)
            new_lines.extend(new_side)
        text = "\n".join(new_lines)
        if new_lines:
            text += "\n"
        _atomic_write_text(path, text)
        return

    if not path.is_file():
        raise FileNotFoundError(
            f"patch targets missing file: {path.relative_to(WORKSPACE.resolve())}"
        )

    content = path.read_text(encoding="utf-8", errors="replace")
    file_lines = content.splitlines(keepends=False)
    ended_with_newline = not content or content.endswith("\n")

    sorted_hunks = sorted(hunks, key=lambda h: h.old_start, reverse=True)
    for h in sorted_hunks:
        old_side, new_side = _split_hunk_sides(h.raw_body)
        start = h.old_start - 1
        end = start + h.old_count
        if start < 0 or end > len(file_lines):
            rel = path.relative_to(WORKSPACE.resolve())
            raise ValueError(
                f"hunk @@ -{h.old_start},{h.old_count} @@ out of range for "
                f"{rel.as_posix()} (file has {len(file_lines)} lines)"
            )
        actual = file_lines[start:end]
        if actual != old_side:
            rel = path.relative_to(WORKSPACE.resolve())
            raise ValueError(
                f"hunk context mismatch in {rel.as_posix()} at line {h.old_start}: "
                f"expected {old_side!r}, found {actual!r}"
            )
        file_lines[start:end] = new_side

    new_content = "\n".join(file_lines)
    if ended_with_newline:
        new_content += "\n"
    _atomic_write_text(path, new_content)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                log.warning("could not remove temp file %s", tmp)


def apply_unified_patch(unified_diff: str) -> str:
    """Apply a unified diff touching paths under ``WORKSPACE``. Returns a short human summary."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ws = WORKSPACE.resolve()
    text = unified_diff.replace("\r\n", "\n")
    raw_lines = text.split("\n")

    idx = 0
    while idx < len(raw_lines) and not raw_lines[idx].startswith("--- "):
        idx += 1

    reports: list[str] = []
    while idx < len(raw_lines):
        if not raw_lines[idx].startswith("--- "):
            idx += 1
            continue

        old_raw, new_raw, hunks, idx = _parse_file_segment(raw_lines, idx)

        old_path = _normalize_diff_path(old_raw) if old_raw != "/dev/null" else None
        new_path = _normalize_diff_path(new_raw) if new_raw != "/dev/null" else None

        if old_path and new_path:
            opath = _workspace_target(old_path)
            npath = _workspace_target(new_path)
            if opath != npath:
                raise ValueError(
                    f"rename not supported ({old_path!r} -> {new_path!r}); "
                    "use same path in --- and +++ for edits"
                )
            target = opath
            is_new = False
        elif old_path is None and new_path:
            target = _workspace_target(new_path)
            is_new = True
        elif old_path and new_path is None:
            target = _workspace_target(old_path)
            if target.is_file():
                target.unlink()
            reports.append(f"deleted {target.relative_to(ws).as_posix()}")
            continue
        else:
            raise ValueError("invalid ---/+++ pair in patch")

        _apply_one_file(target, hunks, is_new=is_new)
        rel = target.relative_to(ws).as_posix()
        reports.append(f"created {rel}" if is_new else f"updated {rel}")

    if not reports:
        return "no file operations in patch (no --- hunks found)"
    return "; ".join(reports)


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "apply_unified_patch",
            "description": (
                "Apply a git-style unified diff to workspace files. Use ---/+++ paths relative to "
                "workspace (git prefixes a/ b/ are OK). One call can touch multiple files. "
                "Use shell_exec (cat, head, sed, etc.) to read the current file before generating the "
                "diff so hunks match. On context mismatch, re-read and regenerate the patch. "
                "Renames (different --- vs +++ paths) are not supported."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "unified_diff": {
                        "type": "string",
                        "description": "Full unified diff body (---, +++, @@ hunks)",
                    },
                },
                "required": ["unified_diff"],
            },
        },
    },
]

FUNCTIONS = {
    "apply_unified_patch": apply_unified_patch,
}
