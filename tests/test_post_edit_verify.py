"""Post-edit verification (ruff / optional pytest) after workspace_search_replace."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import config
import tools.post_edit_verify as pe


def test_append_skips_when_not_updated() -> None:
    assert pe.append_workspace_edit_verification("x.py", "Error in workspace_search_replace: nope") == (
        "Error in workspace_search_replace: nope"
    )
    assert pe.append_workspace_edit_verification("x.py", "no change") == "no change"


def test_append_skips_when_both_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "AGENT_POST_EDIT_VERIFY", False)
    monkeypatch.setattr(config, "AGENT_POST_EDIT_PYTEST", False)
    assert pe.append_workspace_edit_verification("z.py", "updated a/b.py") == "updated a/b.py"


def test_pytest_candidates_matches_test_stem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "tests").mkdir(parents=True)
    mod = tmp_path / "mymod.py"
    mod.write_text("x = 1\n", encoding="utf-8")
    testp = tmp_path / "tests" / "test_mymod.py"
    testp.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(pe, "_REPO_ROOT", tmp_path)
    cands = pe._pytest_candidates(mod)
    assert testp in cands


def test_append_ruff_section_when_ruff_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "AGENT_POST_EDIT_VERIFY", True)
    monkeypatch.setattr(config, "AGENT_POST_EDIT_PYTEST", False)
    p = tmp_path / "mod.py"
    p.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(pe, "_workspace_target", lambda rel: p)
    monkeypatch.setattr(pe, "_REPO_ROOT", tmp_path)

    def fake_run(*_a, **_k):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="mod.py:1:1: E999\n")

    monkeypatch.setattr(pe.subprocess, "run", fake_run)
    monkeypatch.setattr(pe.shutil, "which", lambda name: "/bin/ruff" if name == "ruff" else None)

    out = pe.append_workspace_edit_verification("mod.py", "updated workspace/mod.py")
    assert "post-edit" in out
    assert "ruff check" in out
    assert "E999" in out


def test_append_pytest_skip_message_when_no_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "AGENT_POST_EDIT_VERIFY", False)
    monkeypatch.setattr(config, "AGENT_POST_EDIT_PYTEST", True)
    p = tmp_path / "orphan.py"
    p.write_text("# no matching tests/\n", encoding="utf-8")
    monkeypatch.setattr(pe, "_workspace_target", lambda rel: p)
    monkeypatch.setattr(pe, "_REPO_ROOT", tmp_path)

    out = pe.append_workspace_edit_verification("orphan.py", "updated workspace/orphan.py")
    assert "pytest" in out
    assert "skipped" in out
