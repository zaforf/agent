"""Shell execution tool — single long-lived bash process in a workspace directory.

One shared subprocess.Popen(bash) is started on first use and reused across calls.
State (environment variables, cwd, installed packages) persists between calls.
stderr is merged into stdout so the model sees error messages inline.

The shell starts in WORKSPACE (AGENT_WORKSPACE env var, default <project>/workspace).
Files written there survive server restarts because the directory is on the host
filesystem (bind-mount it in production so files also survive container rebuilds).

Large outputs (>8 000 chars) are automatically summarized by the agent loop before
being stored in history — no special handling needed here.

All calls are serialized via threading.Lock (single-chat deployment; belt-and-suspenders).
Classified as a blocking sync tool in agent.py — runs via asyncio.to_thread.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

WORKSPACE: Path = Path(
    os.environ.get("AGENT_WORKSPACE", str(Path(__file__).parent.parent / "workspace"))
).resolve()

_DEFAULT_TIMEOUT: int = int(os.environ.get("AGENT_SHELL_TIMEOUT", "30"))

# 1 MB hard cap — if a command produces more than this, kill and restart the shell
# rather than buffering indefinitely (guards against cat /dev/urandom etc.).
_MAX_OUTPUT_BYTES: int = 1_000_000

# Unique per-process sentinel used to delimit command output.
# The 32-char hex UUID makes accidental collision in command output negligible.
_SENTINEL: str = f"__DONE_{uuid.uuid4().hex}__"

_proc: subprocess.Popen | None = None
_out_queue: queue.Queue = queue.Queue()
_lock: threading.Lock = threading.Lock()


def _start_shell() -> None:
    global _proc, _out_queue

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    _out_queue = queue.Queue()

    _proc = subprocess.Popen(
        ["/bin/bash"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(WORKSPACE),
        env=dict(os.environ),
        bufsize=0,
    )
    log.info("shell: started pid=%d workspace=%s", _proc.pid, WORKSPACE)

    def _reader() -> None:
        # Capture both references at thread creation time — a later _start_shell()
        # sets _proc = None then creates a new process; accessing the global on
        # each iteration races with that None assignment (AttributeError).
        q    = _out_queue
        proc = _proc
        try:
            while True:
                chunk = proc.stdout.readline()
                if not chunk:
                    q.put(None)  # EOF — shell exited
                    break
                q.put(chunk)
        except Exception as exc:
            log.warning("shell: reader error: %s", exc)
            q.put(None)

    threading.Thread(target=_reader, daemon=True, name="shell-reader").start()


def _ensure_shell() -> None:
    global _proc
    if _proc is None or _proc.poll() is not None:
        if _proc is not None:
            log.warning("shell: process exited (code=%s), restarting", _proc.poll())
        _start_shell()


def _kill_shell() -> None:
    global _proc
    if _proc is not None:
        try:
            _proc.kill()
            _proc.wait(timeout=2)
        except Exception:
            pass
        _proc = None


def shell_exec(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute a shell command and return combined stdout + stderr.

    The bash session persists across calls — environment variables, working
    directory, and installed packages carry over. The shell starts in WORKSPACE.
    On timeout or unexpected death the shell is restarted; workspace files are
    unaffected.

    timeout: seconds to wait for the command (default from AGENT_SHELL_TIMEOUT,
    typically 30). Increase for long-running tasks like package installs or builds.
    """
    if not command or not command.strip():
        return "(empty command)"

    with _lock:
        _ensure_shell()

        # Drain any stale output. Normally the queue is empty here (we consumed
        # all output up to the sentinel on the previous call, and _start_shell
        # creates a fresh queue). This is belt-and-suspenders.
        while not _out_queue.empty():
            try:
                _out_queue.get_nowait()
            except queue.Empty:
                break

        # Append sentinel + exit-code capture after the user's command.
        # The sentinel echo always runs because it is a separate statement;
        # even if the user's command fails, we learn the exit code.
        payload = command.rstrip("\n") + f'\n__rc=$?; echo "{_SENTINEL}:$__rc"\n'
        try:
            _proc.stdin.write(payload.encode())
            _proc.stdin.flush()
        except BrokenPipeError:
            log.warning("shell: broken pipe, restarting")
            _kill_shell()
            _start_shell()
            _proc.stdin.write(payload.encode())
            _proc.stdin.flush()

        lines: list[str] = []
        total_bytes: int = 0
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("shell: command timed out after %ds", timeout)
                _kill_shell()
                partial = "".join(lines).rstrip("\n")
                trailer = f"[timed out after {timeout}s — output may be incomplete; shell restarted]"
                return f"{partial}\n{trailer}" if partial else trailer

            try:
                raw = _out_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue

            if raw is None:
                _kill_shell()
                return "Error: shell process died unexpectedly"

            decoded = raw.decode("utf-8", errors="replace")

            if _SENTINEL in decoded:
                try:
                    exit_code = int(decoded.strip().rsplit(":", 1)[-1])
                except (ValueError, IndexError):
                    exit_code = 0
                output = "".join(lines).rstrip("\n")
                if exit_code != 0:
                    trailer = f"(exit code {exit_code})"
                    output = f"{output}\n{trailer}" if output else trailer
                return output or "(no output)"

            total_bytes += len(raw)
            if total_bytes > _MAX_OUTPUT_BYTES:
                log.warning("shell: output cap exceeded (%d bytes), restarting", total_bytes)
                _kill_shell()
                output = "".join(lines)[:_MAX_OUTPUT_BYTES].rstrip("\n")
                return (
                    output
                    + f"\n[output truncated — exceeded {_MAX_OUTPUT_BYTES // 1024}KB limit; shell restarted]"
                )

            lines.append(decoded)


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": (
                "Execute a shell command in a persistent bash session and return combined stdout+stderr. "
                "The session persists across calls — environment variables, working directory changes, "
                "and installed packages carry over between commands. "
                "The shell starts in the workspace directory (AGENT_WORKSPACE env var); "
                "files written there survive server restarts. "
                "Network is available (curl, pip, git, etc.). "
                "Use for running scripts, installing packages, file I/O, compiling code, and automation. "
                "Interactive commands (editors, pagers, prompts) will time out — "
                "use non-interactive flags instead (e.g. pip install -q, apt-get install -y, "
                "python -c '...', git -C /path ...)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run. Multi-line commands are supported.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Seconds to wait before killing the command "
                            f"(default {_DEFAULT_TIMEOUT}). "
                            "Increase for long-running tasks like package installs or builds."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    }
]

FUNCTIONS = {"shell_exec": shell_exec}
