"""Language-server navigation (Pyright) — go-to-definition, references, symbols.

Uses JSON-RPC over stdio against the ``pyright-langserver`` entrypoint shipped
inside the npm **``pyright``** package (default argv:
``npx -y --package=pyright pyright-langserver --stdio`` — there is no standalone
``pyright-langserver`` package on npm). **Python-first:** Pyright is invoked with
the shell's current working directory as ``rootUri``, matching ``workspace_read``
/ ``workspace_grep`` path resolution.

Environment:

- ``LSP_PYRIGHT_COMMAND`` — JSON array of argv tokens for the server process
  (default includes ``--package=pyright`` so npx resolves the correct binary).

Outputs are capped (references, outline size) so tool results stay bounded for
the model and history summarization.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from tools.shell import WORKSPACE, get_shell_cwd
from tools.workspace_patch import _workspace_target

log = logging.getLogger(__name__)

# ── caps (keep tool output bounded) ───────────────────────────────────────────
_MAX_REFERENCES: int = 48
_MAX_WORKSPACE_SYMBOLS: int = 80
_MAX_OUTLINE_LINES: int = 220

# Default: npx runs the ``pyright-langserver`` binary from the ``pyright`` npm package
# (there is no standalone ``pyright-langserver`` package on npm).
_DEFAULT_CMD_JSON = '["npx","-y","--package=pyright","pyright-langserver","--stdio"]'


def _pyright_command() -> list[str]:
    raw = os.environ.get("LSP_PYRIGHT_COMMAND", _DEFAULT_CMD_JSON).strip()
    try:
        cmd = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LSP_PYRIGHT_COMMAND must be a JSON array of strings: {e}") from e
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        raise ValueError("LSP_PYRIGHT_COMMAND must be a JSON array of strings")
    if not cmd:
        raise ValueError("LSP_PYRIGHT_COMMAND is empty")
    return cmd


def _grep_style_to_lsp_position(line_1based: int, column_1based: int) -> dict[str, int]:
    """Convert 1-based line/column (grep / workspace_read style) to LSP 0-based."""
    ln = max(1, int(line_1based)) - 1
    col = max(1, int(column_1based)) - 1
    return {"line": ln, "character": col}


def _path_to_uri(p: Path) -> str:
    return p.resolve().as_uri()


def _uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    raw = uri[7:]
    if raw.startswith("//"):
        raw = raw[2:]
    try:
        return Path(raw).resolve()
    except OSError:
        return None


def _under_workspace(p: Path) -> bool:
    try:
        p.resolve().relative_to(WORKSPACE.resolve())
        return True
    except ValueError:
        return False


def _safe_rel_path(uri: str) -> str | None:
    path = _uri_to_path(uri)
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(WORKSPACE.resolve()))
    except ValueError:
        return None


class _PyrightJsonRpc:
    """Minimal LSP client: Content-Length framing, synchronous request/response."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._next_id = 1

    def close(self) -> None:
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def _write_body(self, obj: dict[str, Any]) -> None:
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        assert self._proc.stdin is not None
        self._proc.stdin.write(header + data)
        self._proc.stdin.flush()

    def _read_one_message(self) -> dict[str, Any] | None:
        assert self._proc.stdout is not None
        hdr = b""
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            hdr += line
        m = re.search(rb"Content-Length:\s*(\d+)", hdr, re.I)
        if not m:
            log.warning("lsp: malformed header %r", hdr[:200])
            return None
        n = int(m.group(1))
        body = self._proc.stdout.read(n)
        if len(body) < n:
            log.warning("lsp: short read body=%d expected=%d", len(body), n)
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            log.warning("lsp: invalid json in body")
            return None

    def request(self, method: str, params: dict[str, Any] | None) -> Any:
        with self._write_lock, self._read_lock:
            rid = self._next_id
            self._next_id += 1
            payload: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
            if method == "shutdown" and params is None:
                payload["params"] = None
            else:
                payload["params"] = params if params is not None else {}
            self._write_body(payload)
            while True:
                msg = self._read_one_message()
                if msg is None:
                    raise RuntimeError("LSP process ended unexpectedly (no message)")
                if msg.get("id") == rid:
                    if "error" in msg:
                        err = msg["error"]
                        raise RuntimeError(f"LSP error {err.get('code')}: {err.get('message')}")
                    return msg.get("result")
                # notification — ignore (e.g. window/logMessage, textDocument/publishDiagnostics)

    def notify(self, method: str, params: dict[str, Any] | None) -> None:
        with self._write_lock:
            body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                body["params"] = params
            self._write_body(body)


class _PyrightSession:
    """One Pyright subprocess rooted at ``root`` (shell cwd)."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        cmd = _pyright_command()
        log.info("lsp: starting pyright cmd=%s root=%s", cmd, self.root)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.root),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            bufsize=0,
        )
        try:
            if self._proc.stdin is None or self._proc.stdout is None:
                raise RuntimeError("LSP subprocess missing stdio pipes")
            self.rpc = _PyrightJsonRpc(self._proc)
            root_uri = _path_to_uri(self.root)
            self.rpc.request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": root_uri,
                    "capabilities": {
                        "workspace": {"symbol": {"dynamicRegistration": False}},
                    },
                    "initializationOptions": {},
                },
            )
            self.rpc.notify("initialized", {})
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
            raise

    def shutdown(self) -> None:
        try:
            self.rpc.request("shutdown", None)
        except Exception as e:
            log.debug("lsp shutdown: %s", e)
        try:
            self.rpc.notify("exit", None)
        except Exception:
            pass
        self.rpc.close()

    def _file_uri(self, rel: str) -> str:
        return _path_to_uri(_workspace_target(rel))

    def did_open(self, rel: str) -> None:
        target = _workspace_target(rel)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {rel!r}")
        text = target.read_text(encoding="utf-8", errors="replace")
        ext = target.suffix.lower()
        lang = "python" if ext in (".py", ".pyi") else "plaintext"
        self.rpc.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": _path_to_uri(target),
                    "languageId": lang,
                    "version": 1,
                    "text": text,
                }
            },
        )

    def definition(self, rel: str, line: int, column: int) -> Any:
        self.did_open(rel)
        uri = self._file_uri(rel)
        pos = _grep_style_to_lsp_position(line, column)
        return self.rpc.request(
            "textDocument/definition",
            {"textDocument": {"uri": uri}, "position": pos},
        )

    def references(self, rel: str, line: int, column: int, include_declaration: bool) -> Any:
        self.did_open(rel)
        uri = self._file_uri(rel)
        pos = _grep_style_to_lsp_position(line, column)
        return self.rpc.request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": pos,
                "context": {"includeDeclaration": bool(include_declaration)},
            },
        )

    def document_symbol(self, rel: str) -> Any:
        self.did_open(rel)
        uri = self._file_uri(rel)
        return self.rpc.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})

    def workspace_symbol(self, query: str) -> Any:
        return self.rpc.request("workspace/symbol", {"query": query})


_SESSION_LOCK = threading.Lock()
_SESSION: _PyrightSession | None = None
_SESSION_ROOT: Path | None = None


def _get_session() -> _PyrightSession:
    global _SESSION, _SESSION_ROOT
    cwd = get_shell_cwd().resolve()
    with _SESSION_LOCK:
        if _SESSION is not None and _SESSION_ROOT == cwd:
            return _SESSION
        if _SESSION is not None:
            try:
                _SESSION.shutdown()
            except Exception as e:
                log.warning("lsp: old session shutdown: %s", e)
            _SESSION = None
            _SESSION_ROOT = None
        try:
            _SESSION = _PyrightSession(cwd)
        except FileNotFoundError as e:
            raise RuntimeError(
                "Could not start pyright-langserver (executable not found). "
                "Install Node.js so `npx` is available, or set LSP_PYRIGHT_COMMAND "
                "to a JSON argv array (default uses npx --package=pyright). "
                f"Underlying: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Pyright LSP failed to start: {e}. "
                "Try: npx -y --package=pyright pyright-langserver --stdio"
            ) from e
        _SESSION_ROOT = cwd
        return _SESSION


def _format_range(loc: dict[str, Any]) -> str:
    r = loc.get("range") or {}
    start = r.get("start") or {}
    ln = int(start.get("line", 0)) + 1
    ch = int(start.get("character", 0)) + 1
    return f"{ln}:{ch}"


def _format_location_block(uri: str, loc: dict[str, Any], note: str = "") -> str:
    rel = _safe_rel_path(uri)
    if rel:
        pos = _format_range(loc)
        base = f"- `{rel}` @ {pos}"
    else:
        base = "- (definition outside workspace — stdlib/site-packages; use import name or docs)"
    return f"{base} {note}".rstrip()


def _locations_from_result(result: Any) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if result is None:
        return out
    if isinstance(result, dict) and "uri" in result:
        return [(str(result["uri"]), result)]
    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            if "uri" in item and "range" in item:
                out.append((str(item["uri"]), item))
            elif "targetUri" in item:
                uri = str(item["targetUri"])
                r = item.get("targetSelectionRange") or item.get("targetRange") or {}
                out.append((uri, {"uri": uri, "range": r}))
    return out


def _flatten_document_symbols(
    nodes: list[dict[str, Any]],
    rel_path: str,
    depth: int = 0,
    out_lines: list[str] | None = None,
) -> list[str]:
    if out_lines is None:
        out_lines = []
    if len(out_lines) >= _MAX_OUTLINE_LINES:
        return out_lines
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("name", "?")
        kind = node.get("kind", "")
        detail = node.get("detail") or ""
        rng = node.get("selectionRange") or node.get("range") or {}
        st = (rng.get("start") or {})
        line = int(st.get("line", 0)) + 1
        pad = "  " * depth
        tail = f" — {detail}" if detail else ""
        out_lines.append(f"{pad}- {name} (kind={kind}) @ `{rel_path}`:{line}{tail}")
        if len(out_lines) >= _MAX_OUTLINE_LINES:
            break
        children = node.get("children")
        if isinstance(children, list) and children:
            _flatten_document_symbols(children, rel_path, depth + 1, out_lines)
    return out_lines


def lsp_go_to_definition(path: str, line: int, column: int = 1) -> str:
    """Jump to the definition of the symbol at (line, column) in a workspace file.

    ``line`` and ``column`` are **1-based**, matching ``workspace_grep`` /
    ``workspace_read`` line prefixes. Default ``column=1`` selects the start
    of the line (works when the cursor is on the ``def`` / ``class`` keyword).

    **Python-first:** powered by Pyright. Non-Python files may return no results.
    """
    try:
        s = _get_session()
        result = s.definition(path.strip(), int(line), int(column))
    except Exception as e:
        return f"Error in lsp_go_to_definition: {e}"
    locs = _locations_from_result(result)
    if not locs:
        return "No definition found (try a different line/column on the symbol name)."
    lines = ["**Definitions**"]
    for uri, loc in locs[:12]:
        lines.append(_format_location_block(uri, loc))
    if len(locs) > 12:
        lines.append(f"(… {len(locs) - 12} more locations omitted)")
    return "\n".join(lines)


def lsp_find_references(path: str, line: int, column: int = 1, include_declaration: bool = True) -> str:
    """List workspace references to the symbol at (1-based line, column).

    Results are capped and paths are restricted to the agent workspace mount.
    """
    try:
        s = _get_session()
        result = s.references(path.strip(), int(line), int(column), include_declaration)
    except Exception as e:
        return f"Error in lsp_find_references: {e}"
    locs = _locations_from_result(result)
    ws_locs: list[tuple[str, dict[str, Any]]] = []
    for uri, loc in locs:
        if _safe_rel_path(uri):
            ws_locs.append((uri, loc))
    if not ws_locs:
        return "No in-workspace references found."
    lines = [f"**References** (showing up to {_MAX_REFERENCES}, workspace-only)"]
    for uri, loc in ws_locs[:_MAX_REFERENCES]:
        lines.append(_format_location_block(uri, loc))
    if len(ws_locs) > _MAX_REFERENCES:
        lines.append(f"(… {len(ws_locs) - _MAX_REFERENCES} more omitted)")
    return "\n".join(lines)


def lsp_outline(path: str) -> str:
    """Structured outline (functions/classes) for one file — without reading the whole file."""
    try:
        s = _get_session()
        result = s.document_symbol(path.strip())
    except Exception as e:
        return f"Error in lsp_outline: {e}"
    if not result:
        return "No symbols (empty file, unsupported type, or Pyright returned nothing)."
    if not isinstance(result, list):
        return f"Unexpected documentSymbol shape: {type(result).__name__}"
    rel = path.strip()
    lines = _flatten_document_symbols(result, rel)
    if not lines:
        return "No symbols parsed from Pyright response."
    hdr = f"**Outline** for `{path.strip()}` (cap {_MAX_OUTLINE_LINES} lines)\n"
    body = "\n".join(lines)
    if len(lines) >= _MAX_OUTLINE_LINES:
        body += "\n(… outline truncated — use workspace_read on a narrow range if needed)"
    return hdr + body


def lsp_workspace_symbols(query: str) -> str:
    """Fuzzy / name query across the workspace (``workspace/symbol``).

    Prefer a concrete substring (e.g. ``Summarize``, ``run_stream``). Min length 2.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return "Error: query must be at least 2 characters."
    try:
        s = _get_session()
        result = s.workspace_symbol(q)
    except Exception as e:
        return f"Error in lsp_workspace_symbols: {e}"
    if not result:
        return f"No workspace symbols matching {q!r}."
    lines = [f"**Workspace symbols** matching {q!r} (up to {_MAX_WORKSPACE_SYMBOLS}, workspace-only)"]
    n = 0
    for item in result:
        if not isinstance(item, dict):
            continue
        loc = item.get("location") or {}
        uri = str(loc.get("uri") or "")
        if not _safe_rel_path(uri):
            continue
        name = item.get("name", "?")
        kind = item.get("kind", "")
        lines.append(_format_location_block(uri, loc) + f" — `{name}` kind={kind}")
        n += 1
        if n >= _MAX_WORKSPACE_SYMBOLS:
            break
    if n == 0:
        return f"No in-workspace symbols for {q!r} (matches may be in dependencies only)."
    return "\n".join(lines)


SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lsp_go_to_definition",
            "description": (
                "Pyright LSP: jump to where the symbol at (line, column) is defined. "
                "Use after workspace_grep / workspace_read to land on a name. "
                "Line and column are 1-based like grep/read output; default column=1 (start of line). "
                "Python-first; requires Node+npx and the pyright npm package (see LSP_PYRIGHT_COMMAND)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "line": {"type": "integer", "description": "1-based line number"},
                    "column": {
                        "type": "integer",
                        "description": "1-based UTF-16 column (default 1 = line start)",
                        "default": 1,
                    },
                },
                "required": ["path", "line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_find_references",
            "description": (
                "Pyright LSP: list references to the symbol at (line, column) within the workspace. "
                "1-based positions. Results capped; use a tight symbol location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "column": {"type": "integer", "default": 1},
                    "include_declaration": {
                        "type": "boolean",
                        "description": "Include the definition site in results",
                        "default": True,
                    },
                },
                "required": ["path", "line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_outline",
            "description": (
                "Pyright LSP: structured list of functions/classes/etc. in one file "
                "(document symbols) without reading the entire file body."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_workspace_symbols",
            "description": (
                "Pyright LSP: search symbols across the indexed workspace (fuzzy-ish name query). "
                "Good for 'where is X implemented' before opening files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Symbol name substring (min 2 chars)"},
                },
                "required": ["query"],
            },
        },
    },
]

FUNCTIONS: dict[str, callable] = {
    "lsp_go_to_definition": lsp_go_to_definition,
    "lsp_find_references": lsp_find_references,
    "lsp_outline": lsp_outline,
    "lsp_workspace_symbols": lsp_workspace_symbols,
}
