"""Post-edit verification (ruff / optional pytest) after workspace_search_replace."""

from __future__ import annotations

import json
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

    def fake_run(argv, *_a, **_k):
        if "--output-format=json" in argv:
            diag = {
                "cell": None,
                "code": "E999",
                "message": "boom",
                "filename": str(p),
                "location": {"row": 1, "column": 1},
                "end_location": {"row": 1, "column": 2},
                "fix": None,
                "noqa_row": None,
                "severity": "error",
                "url": "",
            }
            return types.SimpleNamespace(returncode=1, stdout=json.dumps([diag]), stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="mod.py:1:1: E999\n")

    monkeypatch.setattr(pe.subprocess, "run", fake_run)
    monkeypatch.setattr(pe.shutil, "which", lambda name: "/bin/ruff" if name == "ruff" else None)

    out = pe.append_workspace_edit_verification("mod.py", "updated workspace/mod.py")
    assert "post-edit" in out
    assert "ruff check" in out
    assert "E999" in out


def test_slice_ruff_edit_span_multiline_remove_import() -> None:
    src = "import sys\nimport os\nx = 1\n"
    old = pe._slice_ruff_edit_span(
        src,
        {"row": 1, "column": 1},
        {"row": 2, "column": 1},
    )
    assert old == "import sys\n"


def test_slice_ruff_edit_span_same_line() -> None:
    src = "x==None\n"
    old = pe._slice_ruff_edit_span(
        src,
        {"row": 1, "column": 1},
        {"row": 1, "column": 8},
    )
    assert old == "x==None"


def test_format_ruff_summary_over_threshold(tmp_path: Path) -> None:
    diags = []
    for i in range(11):
        diags.append(
            {
                "code": "F821",
                "message": "Undefined name `log`",
                "filename": str(tmp_path / "a.py"),
                "location": {"row": i + 1, "column": 1},
                "end_location": {"row": i + 1, "column": 2},
                "fix": None,
            }
        )
    text = pe._format_ruff_json_output(diags, repo_root=tmp_path)
    assert "Found 11 issues" in text
    assert "F821: 11" in text
    assert "Details (11 issue(s))" in text


def test_format_ruff_includes_suggested_replace_for_fix(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("import sys\nx = 1\n", encoding="utf-8")
    diags = [
        {
            "code": "F401",
            "message": "`sys` imported but unused",
            "filename": str(p),
            "location": {"row": 1, "column": 8},
            "end_location": {"row": 1, "column": 11},
            "fix": {
                "applicability": "safe",
                "message": "Remove unused import: `sys`",
                "edits": [
                    {
                        "content": "",
                        "location": {"row": 1, "column": 1},
                        "end_location": {"row": 2, "column": 1},
                    }
                ],
            },
        }
    ]
    text = pe._format_ruff_json_output(diags, repo_root=tmp_path)
    assert "[*]" in text
    assert "old_string=" in text
    assert "import sys" in text
    assert "new_string=" in text


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
