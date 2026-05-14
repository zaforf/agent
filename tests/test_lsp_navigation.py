"""LSP navigation tools (Pyright) — unit tests with mocked session."""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.lsp_navigation as lnav


def test_grep_style_to_lsp_position():
    assert lnav._grep_style_to_lsp_position(1, 1) == {"line": 0, "character": 0}
    assert lnav._grep_style_to_lsp_position(10, 5) == {"line": 9, "character": 4}


def test_cap_references_respects_env(monkeypatch):
    monkeypatch.setenv("LSP_MAX_REFERENCES", "10")
    assert lnav._cap_references() == 10


def test_cap_references_env_clamped_high(monkeypatch):
    monkeypatch.setenv("LSP_MAX_REFERENCES", "99999")
    assert lnav._cap_references() == 500


def test_safe_rel_path_under_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(lnav, "WORKSPACE", tmp_path)
    f = tmp_path / "a" / "b.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x", encoding="utf-8")
    uri = f.resolve().as_uri()
    assert lnav._safe_rel_path(uri) == str(f.resolve().relative_to(tmp_path.resolve()))


def test_lsp_outline_mocked_session(monkeypatch):
    class FakeS:
        def document_symbol(self, rel: str):
            assert rel == "pkg/x.py"
            return [
                {
                    "name": "foo",
                    "kind": 12,
                    "range": {
                        "start": {"line": 2, "character": 0},
                        "end": {"line": 10, "character": 1},
                    },
                    "children": [],
                }
            ]

    monkeypatch.setattr(lnav, "_get_session", lambda: FakeS())
    out = lnav.lsp_outline("pkg/x.py")
    assert "Outline" in out
    assert "foo" in out
    assert "`pkg/x.py`:3" in out
    assert "Function" in out


def test_lsp_workspace_symbols_mocked(monkeypatch):
    class FakeS:
        def workspace_symbol(self, query: str):
            assert query == "run"
            root = Path("/tmp/ws").resolve()
            uri = (root / "agent.py").as_uri()
            return [
                {
                    "name": "run_stream",
                    "kind": 12,
                    "location": {
                        "uri": uri,
                        "range": {
                            "start": {"line": 40, "character": 0},
                            "end": {"line": 41, "character": 1},
                        },
                    },
                }
            ]

    root = Path("/tmp/ws").resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.py").write_text("#", encoding="utf-8")
    monkeypatch.setattr(lnav, "WORKSPACE", root)
    monkeypatch.setattr(lnav, "_get_session", lambda: FakeS())
    out = lnav.lsp_workspace_symbols("run")
    assert "run_stream" in out
    assert "agent.py" in out
    assert "Function" in out


def test_lsp_go_to_definition_mocked(monkeypatch):
    class FakeS:
        def definition(self, rel, line, col):
            root = Path("/tmp/ws2").resolve()
            uri = (root / "b.py").as_uri()
            return {"uri": uri, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}}

    root = Path("/tmp/ws2").resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "b.py").write_text("def x(): pass", encoding="utf-8")
    monkeypatch.setattr(lnav, "WORKSPACE", root)
    monkeypatch.setattr(lnav, "_get_session", lambda: FakeS())
    out = lnav.lsp_go_to_definition("b.py", line=1, column=1)
    assert "Definitions" in out
    assert "b.py" in out


@pytest.mark.net
def test_pyright_jsonrpc_roundtrip_smoke():
    """Optional: requires npx + network on first pyright-langserver pull."""
    import shutil
    import subprocess

    if not shutil.which("npx"):
        pytest.skip("npx not installed")
    proc = subprocess.Popen(
        ["npx", "-y", "--package=pyright", "pyright-langserver", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    rpc = None
    try:
        rpc = lnav._PyrightJsonRpc(proc)
        root = Path(__file__).resolve().parent.parent
        rpc.request(
            "initialize",
            {
                "processId": None,
                "rootUri": root.as_uri(),
                "capabilities": {},
            },
        )
        rpc.notify("initialized", {})
    finally:
        if rpc is not None:
            try:
                rpc.request("shutdown", None)
            except Exception:
                pass
            try:
                rpc.notify("exit", None)
            except Exception:
                pass
            rpc.close()
