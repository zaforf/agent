"""Post-edit verification appended to ``workspace_search_replace`` tool results.

Runs ``ruff check`` on the touched ``.py`` file (when ``ruff`` is on ``PATH``) and,
optionally, a **scoped** ``pytest`` invocation for matching test files under
``tests/``. This mirrors a lightweight Cursor-style "edit then see diagnostics"
loop without adding new model-facing tools.

Controlled by ``config.AGENT_POST_EDIT_VERIFY`` and ``config.AGENT_POST_EDIT_PYTEST``.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import config
from tools.workspace_patch import _workspace_target

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_CAP = 8000


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


def append_workspace_edit_verification(path_rel: str, base_result: str) -> str:
    """If ``workspace_search_replace`` succeeded on a ``.py`` file, append ruff/pytest blocks."""
    if not base_result.startswith("updated "):
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
        rbx = shutil.which("ruff")
        if rbx:
            ruff_out = _run_subprocess(
                [rbx, "check", str(target)],
                cwd=_REPO_ROOT,
                timeout=min(timeout, 90),
                label="ruff",
            )
            parts.append("\n\n---\n**post-edit** `ruff check`:\n" + _truncate(ruff_out))
        else:
            parts.append("\n\n---\n**post-edit** `ruff check`: skipped (`ruff` not on PATH)\n")

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
