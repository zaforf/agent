"""Post-edit verification appended to ``workspace_search_replace`` tool results.

Runs ``ruff check`` on the touched ``.py`` file (when ``ruff`` is on ``PATH``) and,
optionally, a **scoped** ``pytest`` invocation for matching test files under
``tests/``. This mirrors a lightweight Cursor-style "edit then see diagnostics"
loop without adding new model-facing tools.

Controlled by ``config.AGENT_POST_EDIT_VERIFY`` and ``config.AGENT_POST_EDIT_PYTEST``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import config
from tools.workspace_patch import _workspace_target

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_CAP = 8000
# When the number of diagnostics reaches this count, prepend a grouped summary (user: >10).
_RUFF_SUMMARY_THRESHOLD = 11
_RUFF_SNIPPET_REPR_CAP = 500


def _truncate(s: str, cap: int = _OUTPUT_CAP) -> str:
    if len(s) <= cap:
        return s
    return s[:cap] + f"\n[… truncated {len(s) - cap} chars]\n"


def _pytest_candidates(abs_py: Path) -> list[Path]:
    """Paths to pass to ``pytest`` for a touched ``.py`` file (existence-filtered)."""
    stem = abs_py.stem
    seen: set[Path] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        rp = p.resolve()
        if rp in seen:
            return
        seen.add(rp)
        if rp.is_file():
            out.append(rp)

    parts = abs_py.parts
    if "tests" in parts and abs_py.suffix.lower() == ".py":
        add(abs_py)
    add(_REPO_ROOT / "tests" / f"test_{stem}.py")
    add(_REPO_ROOT / "tests" / f"{stem}_test.py")
    return out


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    timeout: int,
    label: str,
) -> str:
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
    except subprocess.TimeoutExpired:
        return f"{label}: timed out after {timeout}s"
    except OSError as e:
        return f"{label}: {e}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    merged = "\n".join(x for x in (out, err) if x)
    if proc.returncode == 0 and not merged:
        return f"{label}: OK (exit 0, no output)"
    prefix = "OK" if proc.returncode == 0 else f"exit {proc.returncode}"
    return f"{label}: {prefix}\n{merged}" if merged else f"{label}: {prefix}"


def _ruff_argv_base() -> list[str]:
    """Return argv prefix to invoke Ruff (``ruff`` on PATH, else ``python -m ruff``)."""
    exe = shutil.which("ruff")
    if exe:
        return [exe]
    return [sys.executable, "-m", "ruff"]


def _slice_ruff_edit_span(src: str, loc: dict, end: dict) -> str:
    """Return the source slice replaced by a Ruff JSON ``fix.edits[]`` entry.

    Rows and columns are **1-based**; the end ``(row, column)`` is **exclusive**
    (same convention as Ruff's JSON for ``textDocument``-style ranges).
    """
    if not isinstance(loc, dict) or not isinstance(end, dict):
        return ""
    lines = src.splitlines(keepends=True)
    r1, c1 = int(loc.get("row", 0)), int(loc.get("column", 0))
    r2, c2 = int(end.get("row", 0)), int(end.get("column", 0))
    if r1 < 1 or c1 < 1 or r2 < 1 or c2 < 1:
        return ""
    i0, j0 = r1 - 1, r2 - 1
    if i0 >= len(lines) or j0 < 0:
        return ""
    i0 = min(i0, len(lines) - 1)
    j0 = min(j0, len(lines) - 1)

    if i0 == j0:
        line = lines[i0]
        a, b = c1 - 1, c2 - 1
        a = max(0, min(a, len(line)))
        b = max(0, min(b, len(line)))
        if a > b:
            return ""
        return line[a:b]

    out: list[str] = []
    first = lines[i0]
    out.append(first[c1 - 1 :] if c1 - 1 <= len(first) else "")
    for idx in range(i0 + 1, j0):
        if 0 <= idx < len(lines):
            out.append(lines[idx])
    if j0 < len(lines):
        last = lines[j0][: max(0, c2 - 1)]
        out.append(last)
    return "".join(out)


def _repr_snippet(s: str) -> str:
    r = repr(s)
    if len(r) <= _RUFF_SNIPPET_REPR_CAP:
        return r
    return r[:_RUFF_SNIPPET_REPR_CAP] + f"... (len {len(s)} chars)"


def _format_ruff_json_output(diagnostics: list[dict], *, repo_root: Path) -> str:
    """Turn Ruff ``--output-format=json`` diagnostics into text for the model."""
    if not diagnostics:
        return "ruff: OK (0 issues)\n"

    by_key: Counter[tuple[str, str]] = Counter()
    for d in diagnostics:
        by_key[(str(d.get("code") or "?"), str(d.get("message") or ""))] += 1

    n = len(diagnostics)
    parts: list[str] = []
    if n >= _RUFF_SUMMARY_THRESHOLD:
        parts.append(f"Found {n} issues (summary by code + message):\n")
        for (code, msg), cnt in sorted(by_key.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
            parts.append(f"- {code}: {cnt} — {msg}\n")
        parts.append("\n")

    parts.append(f"Details ({n} issue(s)):\n")

    file_text: dict[str, str] = {}

    def _read_abs(abs_path: Path) -> str:
        key = str(abs_path.resolve())
        if key not in file_text:
            try:
                file_text[key] = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                file_text[key] = ""
        return file_text[key]

    root_res = repo_root.resolve()

    def _sort_key(d: dict) -> tuple:
        loc = d.get("location") or {}
        return (
            str(d.get("filename") or ""),
            int(loc.get("row") or 0),
            int(loc.get("column") or 0),
            str(d.get("code") or ""),
        )

    for d in sorted(diagnostics, key=_sort_key):
        fn = d.get("filename")
        abs_path = Path(str(fn)) if fn else Path()
        try:
            resolved = abs_path.resolve()
            rel = str(resolved.relative_to(root_res)) if resolved.is_relative_to(root_res) else str(resolved)
        except Exception:
            rel = str(fn)

        loc = d.get("location") or {}
        row, col = int(loc.get("row", 0)), int(loc.get("column", 0))
        code = d.get("code") or "?"
        msg = d.get("message") or ""
        fix = d.get("fix")
        star = "[*] " if fix else ""
        parts.append(f"- {star}{rel}:{row}:{col} {code} {msg}\n")

        if not isinstance(fix, dict):
            continue
        applicability = fix.get("applicability")
        edits = fix.get("edits") or []
        if not isinstance(edits, list) or not edits:
            continue
        src = _read_abs(abs_path) if abs_path.is_file() else ""
        for ed in edits:
            if not isinstance(ed, dict):
                continue
            el, ee = ed.get("location") or {}, ed.get("end_location") or {}
            content = ed.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            old_s = _slice_ruff_edit_span(src, el, ee) if src else ""
            app_note = f" [{applicability}]" if applicability else ""
            parts.append(
                f"  suggested replace{app_note}: old_string={_repr_snippet(old_s)} "
                f"new_string={_repr_snippet(content)}\n"
            )

    return "".join(parts)


def _run_ruff_check_pretty(target: Path, repo_root: Path, timeout: int) -> str:
    """Run ``ruff check`` on ``target`` and return formatted output (JSON-based)."""
    argv = _ruff_argv_base() + ["check", str(target), "--output-format=json"]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
    except subprocess.TimeoutExpired:
        return f"ruff: timed out after {timeout}s\n"
    except OSError as e:
        return f"ruff: {e}\n"

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()

    if proc.returncode == 2:
        merged = "\n".join(x for x in (stdout, stderr) if x)
        return f"ruff: exit 2 (configuration or internal error)\n{merged}\n" if merged else (
            "ruff: exit 2 (configuration or internal error)\n"
        )

    if not stdout:
        if proc.returncode == 0:
            return "ruff: OK (0 issues)\n"
        tail = f"\n{stderr}" if stderr else ""
        return f"ruff: exit {proc.returncode} (no JSON on stdout){tail}\n"

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        log.debug("post-edit verify: ruff JSON parse failed, falling back to text output")
        plain_argv = _ruff_argv_base() + ["check", str(target)]
        return _run_subprocess(plain_argv, cwd=repo_root, timeout=timeout, label="ruff")

    if not isinstance(data, list):
        return _run_subprocess(
            _ruff_argv_base() + ["check", str(target)],
            cwd=repo_root,
            timeout=timeout,
            label="ruff",
        )

    if not data and proc.returncode == 0:
        return "ruff: OK (0 issues)\n"

    return _format_ruff_json_output(data, repo_root=repo_root)


def append_workspace_edit_verification(path_rel: str, base_result: str) -> str:
    """If ``workspace_search_replace`` succeeded on a ``.py`` file, append ruff/pytest blocks."""
    if "updated " not in base_result:
        return base_result
    rel = (path_rel or "").strip().replace("\\", "/")
    if not rel:
        return base_result
    try:
        target = _workspace_target(rel)
    except Exception as e:
        log.debug("post-edit verify: skip resolve %r: %s", rel, e)
        return base_result
    if target.suffix.lower() != ".py":
        return base_result

    verify = getattr(config, "AGENT_POST_EDIT_VERIFY", True)
    pytest_en = getattr(config, "AGENT_POST_EDIT_PYTEST", False)
    if not verify and not pytest_en:
        return base_result

    parts: list[str] = [base_result]
    timeout = max(10, int(getattr(config, "AGENT_POST_EDIT_VERIFY_TIMEOUT", 120)))

    if verify:
        # Prefer ``ruff`` on PATH, else ``python -m ruff`` (venv / CI).
        ruff_out = _run_ruff_check_pretty(target, _REPO_ROOT, timeout=min(timeout, 90))
        parts.append("\n\n---\n**post-edit** `ruff check`:\n" + _truncate(ruff_out))

    if pytest_en:
        cands = _pytest_candidates(target)
        if not cands:
            parts.append(
                "\n\n---\n**post-edit** `pytest`: skipped (no `tests/test_<stem>.py` "
                "or `tests/<stem>_test.py` for this file, and file is not under `tests/`)\n"
            )
        else:
            argv = [sys.executable, "-m", "pytest", "-q", "--tb=line", *[str(p) for p in cands]]
            py_out = _run_subprocess(argv, cwd=_REPO_ROOT, timeout=timeout, label="pytest")
            parts.append("\n\n---\n**post-edit** `pytest` (scoped):\n" + _truncate(py_out))

    return "".join(parts)
