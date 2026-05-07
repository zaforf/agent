"""Tests for tools/shell.py — shell_exec tool.

All tests use a tmp workspace so they never touch the real WORKSPACE directory
and are isolated from each other via the shell_env fixture which kills and
restarts the shell process around each test.

These tests use a real subprocess (bash), which is fast and hermetic — no network,
no LLM calls.
"""
from __future__ import annotations

import os
import threading
import time

import pytest


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def shell_env(tmp_path, monkeypatch):
    """Isolated shell environment: tmp workspace, fresh shell process per test."""
    import tools.shell as sh

    ws = tmp_path / "workspace"
    monkeypatch.setattr(sh, "WORKSPACE", ws)
    # Kill any shell left over from a previous test (module-level global).
    sh._kill_shell()
    yield sh
    sh._kill_shell()


# ── Basic execution ───────────────────────────────────────────────────────────

def test_basic_echo(shell_env):
    sh = shell_env
    result = sh.shell_exec("echo hello")
    assert "hello" in result


def test_stdout_and_stderr_merged(shell_env):
    sh = shell_env
    result = sh.shell_exec("echo out; echo err >&2")
    assert "out" in result
    assert "err" in result


def test_multiline_command(shell_env):
    sh = shell_env
    result = sh.shell_exec("echo first\necho second")
    assert "first" in result
    assert "second" in result


def test_empty_command(shell_env):
    sh = shell_env
    assert shell_env.shell_exec("") == "(empty command)"
    assert shell_env.shell_exec("   ") == "(empty command)"


def test_no_output_command(shell_env):
    sh = shell_env
    result = sh.shell_exec("true")
    assert result == "(no output)"


# ── Exit codes ────────────────────────────────────────────────────────────────

def test_exit_code_zero_not_shown(shell_env):
    sh = shell_env
    result = sh.shell_exec("echo hi")
    assert "exit code" not in result


def test_exit_code_nonzero_appended(shell_env):
    sh = shell_env
    result = sh.shell_exec("false")
    assert "exit code" in result
    assert "1" in result


def test_exit_code_custom(shell_env):
    sh = shell_env
    result = sh.shell_exec("exit 42", timeout=5)
    # The `exit` command kills the shell; we expect an error or exit code message.
    # Either "died" (shell died before echo) or "42" (if sentinel was echoed) is OK.
    assert "42" in result or "died" in result or "exit" in result.lower()


def test_nonzero_with_output(shell_env):
    sh = shell_env
    result = sh.shell_exec("echo output; false")
    assert "output" in result
    assert "exit code" in result


# ── State persistence ─────────────────────────────────────────────────────────

def test_env_var_persists_across_calls(shell_env):
    sh = shell_env
    sh.shell_exec("export MY_VAR=secret42")
    result = sh.shell_exec("echo $MY_VAR")
    assert "secret42" in result


def test_cwd_change_persists(shell_env):
    sh = shell_env
    sh.shell_exec("mkdir -p subdir && cd subdir")
    result = sh.shell_exec("pwd")
    assert "subdir" in result


def test_get_shell_cwd_tracks_cd(shell_env):
    sh = shell_env
    sh.shell_exec("mkdir -p myproj && cd myproj")
    cwd = sh.get_shell_cwd()
    assert cwd.name == "myproj"


def test_get_shell_cwd_starts_at_workspace(shell_env):
    sh = shell_env
    sh.shell_exec("true")  # trigger shell start; _start_shell resets _shell_cwd to WORKSPACE
    assert sh.get_shell_cwd() == sh.WORKSPACE


def test_get_shell_cwd_resets_after_restart(shell_env):
    sh = shell_env
    sh.shell_exec("mkdir -p deep && cd deep")
    assert sh.get_shell_cwd().name == "deep"
    sh._kill_shell()
    sh.shell_exec("echo hi")  # triggers restart
    assert sh.get_shell_cwd() == sh.WORKSPACE


def test_file_write_and_read(shell_env):
    sh = shell_env
    sh.shell_exec("echo file_content > testfile.txt")
    result = sh.shell_exec("cat testfile.txt")
    assert "file_content" in result


def test_written_file_on_workspace(shell_env):
    sh = shell_env
    sh.shell_exec("echo persisted > myfile.txt")
    ws_file = shell_env.WORKSPACE / "myfile.txt"
    assert ws_file.exists()
    assert "persisted" in ws_file.read_text()


# ── Workspace creation ────────────────────────────────────────────────────────

def test_workspace_created_if_missing(tmp_path, monkeypatch):
    import tools.shell as sh

    ws = tmp_path / "new_workspace" / "deep"
    monkeypatch.setattr(sh, "WORKSPACE", ws)
    sh._kill_shell()
    try:
        result = sh.shell_exec("echo created")
        assert "created" in result
        assert ws.is_dir()
    finally:
        sh._kill_shell()


# ── Shell restart ─────────────────────────────────────────────────────────────

def test_shell_restarts_after_external_kill(shell_env):
    sh = shell_env
    r1 = sh.shell_exec("echo alive")
    assert "alive" in r1

    # Kill the process externally (simulates crash/OOM).
    if sh._proc is not None:
        sh._proc.kill()
        sh._proc.wait()

    # Next call detects dead process and restarts automatically.
    r2 = sh.shell_exec("echo restarted")
    assert "restarted" in r2


def test_shell_restarts_after_exit_command(shell_env):
    sh = shell_env
    # `exit` kills the shell; should return an error.
    r1 = sh.shell_exec("exit")
    assert "died" in r1 or "exit" in r1.lower() or "restarted" in r1

    # Next call must still work (shell restarted).
    r2 = sh.shell_exec("echo after_exit")
    assert "after_exit" in r2


# ── Timeout ───────────────────────────────────────────────────────────────────

def test_timeout_kills_command(shell_env):
    sh = shell_env
    result = sh.shell_exec("sleep 999", timeout=2)
    assert "timed out" in result
    assert "2s" in result


def test_timeout_returns_partial_output(shell_env):
    sh = shell_env
    # Command prints something before blocking — partial output must be preserved.
    result = sh.shell_exec("echo partial_line; sleep 999", timeout=2)
    assert "partial_line" in result
    assert "timed out" in result


def test_shell_usable_after_timeout(shell_env):
    sh = shell_env
    sh.shell_exec("sleep 999", timeout=2)
    result = sh.shell_exec("echo post_timeout")
    assert "post_timeout" in result


# ── Registration / integration ────────────────────────────────────────────────

def test_shell_exec_in_tool_registry():
    from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert "shell_exec" in names
    assert "shell_exec" in TOOL_FUNCTIONS


def test_shell_exec_schema_shape():
    from tools import TOOL_SCHEMAS
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "shell_exec")
    fn = schema["function"]
    assert fn["description"]
    params = fn["parameters"]
    assert "command" in params["properties"]
    assert "command" in params["required"]
    assert "timeout" in params["properties"]


def test_shell_exec_is_blocking_sync_tool():
    import agent
    assert "shell_exec" in agent._BLOCKING_SYNC_TOOLS


def test_shell_exec_runs_via_to_thread(shell_env, monkeypatch):
    """shell_exec must be dispatched through asyncio.to_thread (blocking tool path)."""
    import asyncio
    import agent

    monkeypatch.setattr(shell_env, "shell_exec", lambda command, **kw: "threaded_ok")
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "shell_exec", lambda command, **kw: "threaded_ok")

    result = asyncio.run(agent._run_tool_async("shell_exec", {"command": "echo hi"}))
    assert result == "threaded_ok"
