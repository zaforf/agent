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

Outputs are capped (references, outline, workspace-symbol rows) with fixed
limits in-module — enough for typical navigation without huge Pyright payloads;
very large tool rows still flow through §6.5 summarization when persisted.
"""
from __future__ import annotations

import json
import keyword
import logging
import os
import re
import subprocess
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.shell import WORKSPACE, get_shell_cwd
from tools.workspace_patch import _workspace_target

log = logging.getLogger(__name__)

# LSP SymbolKind enum (spec 3.17). Official table:
# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#symbolKind
_SYMBOL_KIND_NAMES: dict[int, str] = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}

# Identifiers Pyright won't navigate meaningfully as a "symbol" for def/refs.
_PY_SNAP_SKIP: frozenset[str] = frozenset(
    list(keyword.kwlist) + list(getattr(keyword, "softkwlist", ()))
) | {"True", "False", "None"}

# ``documentSymbol`` kinds to omit from outline (locals / literals — too noisy).
_OUTLINE_OMIT_KINDS: frozenset[int] = frozenset(
    {
        13,  # Variable
        14,  # Constant
        15,  # String
        16,  # Number
        17,  # Boolean
        20,  # Key
        21,  # Null
    }
)

_ANCHOR_DEF_NAME = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)\b")
_ANCHOR_CLASS_NAME = re.compile(r"^[ \t]*class[ \t]+([A-Za-z_]\w*)\b")


def _symbol_kind_name(kind: Any) -> str:
    """Human-readable LSP SymbolKind for tool output."""
    try:
        k = int(kind)
    except (TypeError, ValueError):
        return str(kind) if kind not in (None, "") else "Unknown"
    return _SYMBOL_KIND_NAMES.get(k, f"Kind({k})")


# Output caps (tweak here if needed — keep bounded for latency vs huge LSP trees).
_MAX_REFERENCES: int = 96
_MAX_WORKSPACE_SYMBOLS: int = 120
_MAX_OUTLINE_LINES: int = 400
_MAX_DEFINITION_SNIPPET_LINES: int = 24

# ``workspace/symbol`` buckets: high-signal kinds first, then the rest alphabetically.
_WORKSPACE_SYMBOL_KIND_ORDER: tuple[str, ...] = (
    "Class",
    "Interface",
    "Struct",
    "Function",
    "Method",
    "Constructor",
    "Field",
    "Property",
    "Variable",
    "Constant",
    "Enum",
    "EnumMember",
    "Module",
    "Namespace",
    "Package",
    "TypeParameter",
    "Operator",
    "Event",
    "String",
    "Number",
    "Boolean",
    "Key",
    "Null",
    "File",
    "Array",
    "Object",
)


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


def _safe_rel_path(uri: str) -> str | None:
    path = _uri_to_path(uri)
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(WORKSPACE.resolve()))
    except ValueError:
        return None


def _utf16_codeunits_before(line: str, codepoint_index: int) -> int:
    """UTF-16 length of ``line[:codepoint_index]`` (LSP ``Position.character``)."""
    if codepoint_index <= 0:
        return 0
    if codepoint_index > len(line):
        codepoint_index = len(line)
    return len(line[:codepoint_index].encode("utf-16-le")) // 2


def _identifier_spans(line: str) -> list[tuple[int, int, str]]:
    """``(start, end_exclusive, name)`` for identifiers on the code portion of a line."""
    code = line.split("#", 1)[0]
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\b[A-Za-z_]\w*\b", code)]


def _def_or_class_name_span(line: str) -> tuple[int, int, str] | None:
    """If this line is a ``def`` / ``async def`` / ``class``, return ``(start, end, name)`` for the defined name."""
    m = _ANCHOR_DEF_NAME.match(line)
    if m:
        return (m.start(1), m.end(1), m.group(1))
    m = _ANCHOR_CLASS_NAME.match(line)
    if m:
        return (m.start(1), m.end(1), m.group(1))
    return None


def _snap_identifier_column_1based(line: str, col_1based: int, symbol: str | None) -> tuple[int, str | None]:
    """Pick a 1-based column at an identifier start; return ``(column, picked_name)``."""
    spans = _identifier_spans(line)
    if not spans:
        return col_1based, None
    idx0 = max(0, int(col_1based) - 1)
    col1 = max(1, int(col_1based))
    sym = symbol.strip() if symbol and str(symbol).strip() else None
    if sym:
        exact = [s for s in spans if s[2] == sym]
        if not exact:
            low = sym.lower()
            exact = [s for s in spans if s[2].lower() == low]
        if len(exact) == 1:
            return exact[0][0] + 1, exact[0][2]
        if len(exact) > 1:
            best = min(exact, key=lambda s: min(abs(idx0 - s[0]), abs(idx0 - (s[1] - 1))))
            return best[0] + 1, best[2]

    anchor = _def_or_class_name_span(line)
    if anchor is not None:
        a0, a1, aname = anchor
        # Line-start / indent clicks: jump to the defined name, not ``async``/``def``.
        if col1 == 1 or idx0 < a0:
            return a0 + 1, aname

    for s in spans:
        if s[0] <= idx0 < s[1] and s[2] not in _PY_SNAP_SKIP:
            return s[0] + 1, s[2]

    if anchor is not None:
        a0, a1, aname = anchor
        if idx0 < a1:
            return a0 + 1, aname

    non_kw = [s for s in spans if s[2] not in _PY_SNAP_SKIP]
    pool = non_kw if non_kw else spans
    best = min(pool, key=lambda s: min(abs(idx0 - s[0]), abs(idx0 - (s[1] - 1))))
    return best[0] + 1, best[2]


def _lsp_position_for_anchor(rel: str, line_1: int, col_1: int, symbol: str | None) -> tuple[dict[str, int], str]:
    """0-based LSP ``Position`` plus a short note if we snapped the column."""
    try:
        target = _workspace_target(rel)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines(False)
    except Exception:
        ln = max(0, int(line_1) - 1)
        ch = max(0, int(col_1) - 1)
        return {"line": ln, "character": ch}, ""
    if line_1 < 1 or line_1 > len(lines):
        return {"line": max(0, line_1 - 1), "character": max(0, int(col_1) - 1)}, ""
    text = lines[line_1 - 1]
    col_snapped, picked = _snap_identifier_column_1based(text, col_1, symbol)
    char0 = max(0, col_snapped - 1)
    utf16 = _utf16_codeunits_before(text, char0)
    note = ""
    want = (symbol or "").strip() if symbol else None
    if picked is not None and col_snapped != col_1:
        note = f"Snapped column {col_1} → {col_snapped} (identifier `{picked}`)."
    elif picked is not None and want and (picked == want or picked.lower() == want.lower()):
        note = f"Resolved `{picked}` at column {col_snapped}."
    elif picked is not None:
        note = f"Using identifier `{picked}` at column {col_snapped}."
    return {"line": line_1 - 1, "character": utf16}, note


def _range_start_sort_tuple(loc: dict[str, Any]) -> tuple[int, int]:
    r = loc.get("range") or {}
    st = r.get("start") or {}
    return (int(st.get("line", 0)), int(st.get("character", 0)))


def _snippet_for_location(uri: str, loc: dict[str, Any], max_lines: int) -> str:
    """Numbered source lines from definition start (``workspace_read`` style)."""
    rel = _safe_rel_path(uri)
    if not rel or max_lines <= 0:
        return ""
    try:
        path = _workspace_target(rel)
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines(False)
    except OSError:
        return ""
    rng = loc.get("range") or {}
    st = rng.get("start") or {}
    line0 = int(st.get("line", 0))
    if line0 < 0 or line0 >= len(raw):
        return ""
    chunk = raw[line0 : line0 + max_lines]
    if not chunk:
        return ""
    parts = [f"{line0 + i + 1:>4} | {ln}" for i, ln in enumerate(chunk)]
    return "\n".join(parts)


def _outline_sig_suffix(
    rel_path: str,
    line_1: int,
    kind_label: str,
    detail: str,
    file_lines: list[str] | None,
) -> str:
    # For functions/methods/constructors always show the actual source line.
    # Pyright's detail field for these kinds can contain misleading content
    # (e.g. the module's first import) so we read the file directly instead.
    if kind_label in ("Function", "Method", "Constructor"):
        if file_lines and 1 <= line_1 <= len(file_lines):
            raw = file_lines[line_1 - 1].rstrip()
            if raw:
                if len(raw) > 160:
                    raw = raw[:157] + "..."
                return f" — `{raw}`"
        return ""
    d = (detail or "").strip()
    return f" — {d}" if d else ""


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
            self._warm_workspace()
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
            raise

    _WARM_SKIP_DIRS: frozenset[str] = frozenset({
        ".venv", "venv", "env", ".env",
        "node_modules", "__pycache__", ".git",
        "site-packages", "dist-packages",
    })

    def _warm_workspace(self) -> None:
        """Send didOpen for all project .py files so workspace/symbol has a full index.

        Pyright only returns workspace/symbol results for files it has analysed.
        Sending didOpen notifications (non-blocking) queues them for analysis before
        any workspace/symbol request arrives, so the index is populated in time.
        """
        try:
            py_files: list[Path] = []
            for f in self.root.rglob("*.py"):
                if any(part in self._WARM_SKIP_DIRS for part in f.parts):
                    continue
                py_files.append(f)
                if len(py_files) >= 300:
                    break
            for f in sorted(py_files):
                try:
                    rel = str(f.resolve().relative_to(self.root))
                    self.did_open(rel)
                except Exception:
                    pass
            log.info("lsp: warmed workspace with %d .py files", len(py_files))
        except Exception as e:
            log.warning("lsp: workspace warm-up failed: %s", e)

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

    def definition(self, rel: str, position: dict[str, int]) -> Any:
        self.did_open(rel)
        uri = self._file_uri(rel)
        return self.rpc.request(
            "textDocument/definition",
            {"textDocument": {"uri": uri}, "position": position},
        )

    def references(self, rel: str, position: dict[str, int], include_declaration: bool) -> Any:
        self.did_open(rel)
        uri = self._file_uri(rel)
        return self.rpc.request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": position,
                "context": {"includeDeclaration": bool(include_declaration)},
            },
        )

    def document_symbol(self, rel: str) -> Any:
        self.did_open(rel)
        uri = self._file_uri(rel)
        return self.rpc.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})

    def workspace_symbol(self, query: str) -> Any:
        return self.rpc.request("workspace/symbol", {"query": query})

    def hover(self, rel: str, position: dict[str, int]) -> Any:
        self.did_open(rel)
        uri = self._file_uri(rel)
        return self.rpc.request(
            "textDocument/hover",
            {"textDocument": {"uri": uri}, "position": position},
        )


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
    max_lines: int,
    depth: int = 0,
    out_lines: list[str] | None = None,
    file_lines: list[str] | None = None,
) -> list[str]:
    if out_lines is None:
        out_lines = []
    if len(out_lines) >= max_lines:
        return out_lines
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("name", "?")
        kind = node.get("kind", "")
        try:
            kind_i = int(kind)
        except (TypeError, ValueError):
            kind_i = -1
        detail = node.get("detail") or ""
        rng = node.get("selectionRange") or node.get("range") or {}
        st = (rng.get("start") or {})
        line = int(st.get("line", 0)) + 1
        pad = "  " * depth
        klabel = _symbol_kind_name(kind)
        sig = _outline_sig_suffix(rel_path, line, klabel, detail, file_lines)
        emit = kind_i not in _OUTLINE_OMIT_KINDS
        if emit:
            out_lines.append(f"{pad}{klabel}: {name} @ `{rel_path}`:{line}{sig}")
        if len(out_lines) >= max_lines:
            break
        children = node.get("children")
        if isinstance(children, list) and children:
            _flatten_document_symbols(children, rel_path, max_lines, depth + 1, out_lines, file_lines)
    return out_lines


def lsp_go_to_definition(path: str, line: int, column: int = 1, symbol: str | None = None) -> str:
    """Jump to the definition of the symbol at (line, column) in a workspace file.

    ``line`` and ``column`` are **1-based**, matching ``workspace_grep`` /
    ``workspace_read`` line prefixes. On ``def`` / ``async def`` / ``class`` lines,
    ``column=1`` snaps to the **defined name** (not ``async``/``def``); elsewhere
    the column snaps to a sensible identifier (keywords avoided when possible).

    **Python-first:** powered by Pyright. Non-Python files may return no results.
    """
    rel = path.strip()
    try:
        pos, snap_note = _lsp_position_for_anchor(rel, int(line), int(column), symbol)
        s = _get_session()
        result = s.definition(rel, pos)
    except Exception as e:
        return f"Error in lsp_go_to_definition: {e}"
    locs = _locations_from_result(result)
    if not locs:
        hint = " Optional: pass `symbol` if several names share the line."
        return (
            "No definition found (try `symbol` to pick a name on that line, or a closer line/column)."
            + hint
        )
    lines: list[str] = ["**Definitions**"]
    if snap_note:
        lines.append(snap_note)
    ws_snippets = 0
    for uri, loc in locs[:12]:
        lines.append(_format_location_block(uri, loc))
        if _safe_rel_path(uri) and ws_snippets < 3:
            sn = _snippet_for_location(uri, loc, _MAX_DEFINITION_SNIPPET_LINES)
            if sn:
                lines.append("```text\n" + sn + "\n```")
                ws_snippets += 1
    if len(locs) > 12:
        lines.append(f"(… {len(locs) - 12} more locations omitted)")
    return "\n".join(lines)


def lsp_find_references(
    path: str,
    line: int,
    column: int = 1,
    include_declaration: bool = True,
    symbol: str | None = None,
) -> str:
    """List workspace references to the symbol at (1-based line, column).

    Results are capped and paths are restricted to the agent workspace mount.
    """
    rel = path.strip()
    try:
        pos, snap_note = _lsp_position_for_anchor(rel, int(line), int(column), symbol)
        s = _get_session()
        result = s.references(rel, pos, include_declaration)
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
    if snap_note:
        lines.append(snap_note)
    for uri, loc in ws_locs[:_MAX_REFERENCES]:
        lines.append(_format_location_block(uri, loc))
    if len(ws_locs) > _MAX_REFERENCES:
        lines.append(f"(… {len(ws_locs) - _MAX_REFERENCES} more omitted)")
    return "\n".join(lines)


def lsp_outline(path: str) -> str:
    """Structured outline (functions/classes) for one file — without reading the whole file."""
    rel = path.strip()
    try:
        file_lines = _workspace_target(rel).read_text(encoding="utf-8", errors="replace").splitlines(False)
    except (OSError, ValueError):
        file_lines = None
    try:
        s = _get_session()
        result = s.document_symbol(rel)
    except Exception as e:
        return f"Error in lsp_outline: {e}"
    if not result:
        return "No symbols (empty file, unsupported type, or Pyright returned nothing)."
    if not isinstance(result, list):
        return f"Unexpected documentSymbol shape: {type(result).__name__}"
    lines = _flatten_document_symbols(result, rel, _MAX_OUTLINE_LINES, file_lines=file_lines)
    if not lines:
        return "No symbols parsed from Pyright response."
    hdr = f"**Outline** for `{rel}` (cap {_MAX_OUTLINE_LINES} lines)\n"
    body = "\n".join(lines)
    if len(lines) >= _MAX_OUTLINE_LINES:
        body += "\n(… outline truncated — use workspace_read on a narrow range if needed)"
    return hdr + body


_ID_RE = re.compile(r"^[A-Za-z_]\w*$")
_WARM_SKIP_DIRS_SET: frozenset[str] = frozenset({
    ".venv", "venv", "env", ".env",
    "node_modules", "__pycache__", ".git",
    "site-packages", "dist-packages",
})


def _grep_defs_fallback(query: str) -> list[tuple[str, str, int]]:
    """Pure-Python grep for ``def <query>`` / ``class <query>`` across workspace .py files.

    Returns ``[(kind, rel_path, line_1)]``. Used when Pyright's workspace/symbol
    index hasn't been built yet (e.g. first call before analysis completes).
    """
    pat = re.compile(
        rf"^\s*(?:async\s+)?def\s+({re.escape(query)})\s*[:(]"
        rf"|^\s*class\s+({re.escape(query)})\s*[:(]"
    )
    results: list[tuple[str, str, int]] = []
    try:
        root = WORKSPACE.resolve()
        for f in sorted(root.rglob("*.py")):
            if any(part in _WARM_SKIP_DIRS_SET for part in f.parts):
                continue
            try:
                rel = str(f.resolve().relative_to(root))
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, ValueError):
                continue
            for i, line in enumerate(lines):
                m = pat.match(line)
                if m:
                    kind = "Class" if m.group(2) else "Function"
                    results.append((kind, rel, i + 1))
    except Exception:
        pass
    return results


def lsp_workspace_symbols(query: str) -> str:
    """Fuzzy / name query across the workspace (``workspace/symbol``).

    Prefer a concrete substring (e.g. ``Summarize``, ``run_stream``). Min length 2.
    Falls back to a grep-based def/class search when the LSP index is not yet built.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return "Error: query must be at least 2 characters."
    try:
        s = _get_session()
        result = s.workspace_symbol(q)
    except Exception as e:
        return f"Error in lsp_workspace_symbols: {e}"
    buckets: defaultdict[str, list[tuple[str, str, str, dict[str, Any]]]] = defaultdict(list)
    for item in result or []:
        if not isinstance(item, dict):
            continue
        loc = item.get("location") or {}
        uri = str(loc.get("uri") or "")
        rel2 = _safe_rel_path(uri)
        if not rel2:
            continue
        name = item.get("name", "?")
        kind = item.get("kind", "")
        klabel = _symbol_kind_name(kind)
        buckets[klabel].append((rel2, name, uri, loc))
    if not buckets:
        # LSP index not ready yet (Pyright hasn't finished analysing) — grep fallback.
        if _ID_RE.match(q):
            hits = _grep_defs_fallback(q)
            if hits:
                lines = [f"**Workspace symbols** matching {q!r} (grep fallback — LSP index still warming)"]
                for kind, rel2, line_1 in hits[:_MAX_WORKSPACE_SYMBOLS]:
                    lines.append(f"  {kind}: `{rel2}`:{line_1} — `{q}`")
                return "\n".join(lines)
        return f"No workspace symbols matching {q!r}."
    lines = [f"**Workspace symbols** matching {q!r} (up to {_MAX_WORKSPACE_SYMBOLS}, grouped by kind)"]
    emitted = 0
    ordered = [k for k in _WORKSPACE_SYMBOL_KIND_ORDER if k in buckets]
    rest = sorted(set(buckets) - set(_WORKSPACE_SYMBOL_KIND_ORDER))
    for kind in ordered + rest:
        chunk = sorted(buckets[kind], key=lambda t: (t[0], _range_start_sort_tuple(t[3])))
        lines.append(f"### {kind} ({len(chunk)})")
        for rel2, name, uri, loc in chunk:
            if emitted >= _MAX_WORKSPACE_SYMBOLS:
                lines.append(f"\n(… cap {_MAX_WORKSPACE_SYMBOLS} rows — narrow the query.)")
                return "\n".join(lines)
            lines.append(_format_location_block(uri, loc) + f" — `{name}`")
            emitted += 1
    return "\n".join(lines)


def lsp_hover(path: str, line: int, column: int = 1, symbol: str | None = None) -> str:
    """Return Pyright's type, signature, and documentation for the symbol at (line, column).

    Same coordinate and snapping rules as ``lsp_go_to_definition``. Use this to
    inspect a type, see a function signature with types, or read a docstring without
    opening the defining file.
    """
    rel = path.strip()
    try:
        pos, snap_note = _lsp_position_for_anchor(rel, int(line), int(column), symbol)
        s = _get_session()
        result = s.hover(rel, pos)
    except Exception as e:
        return f"Error in lsp_hover: {e}"
    if not result:
        return "No hover information at that position."
    contents = result.get("contents") or ""
    if isinstance(contents, str):
        text = contents
    elif isinstance(contents, dict):
        text = contents.get("value") or contents.get("text") or ""
    elif isinstance(contents, list):
        parts: list[str] = []
        for c in contents:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                parts.append(c.get("value") or c.get("text") or "")
        text = "\n\n".join(p for p in parts if p)
    else:
        text = str(contents)
    text = (text or "").strip()
    if not text:
        return "No hover information at that position."
    out = [f"**Hover** `{rel}`:{line}"]
    if snap_note:
        out.append(snap_note)
    out.append(text)
    return "\n".join(out)


SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lsp_go_to_definition",
            "description": (
                "Python: jump from a position in a file to where that name is defined. "
                "Use 1-based line and column like `workspace_grep` / `workspace_read` line labels. "
                "On `def` / `async def` / `class` lines, `column=1` targets the defined name; "
                "otherwise the tool snaps to a nearby non-keyword identifier. Optional `symbol` "
                "disambiguates busy lines. Returns a short source snippet for in-workspace definitions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "line": {"type": "integer", "description": "1-based line number"},
                    "column": {
                        "type": "integer",
                        "description": "1-based column on that line (default 1 = line start)",
                        "default": 1,
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Optional exact identifier on that line (disambiguation)",
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
                "Python: list references in the workspace to the symbol at (1-based line, column). "
                "Same snapping rules as `lsp_go_to_definition` (`def`/`class` line column 1 → defined name; "
                "optional `symbol`). Results are capped."
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
                    "symbol": {
                        "type": "string",
                        "description": "Optional exact identifier on that line (disambiguation)",
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
                "Python: structured outline (classes, functions, methods, fields, etc.) for one `.py` "
                "file with line numbers—without reading the whole file. Local variables and literal "
                "symbol kinds are omitted to reduce noise."
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
                "Python: search symbols by name substring across the project (min 2 characters). "
                "Use to find likely files before opening them. Falls back to grep when the LSP "
                "index is still warming up."
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
    {
        "type": "function",
        "function": {
            "name": "lsp_hover",
            "description": (
                "Python: return Pyright's type signature and documentation for the symbol at "
                "(1-based line, column). Use to inspect a type, read a function signature with "
                "full type annotations, or view a docstring — without opening the defining file. "
                "Same coordinate and snapping rules as `lsp_go_to_definition`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "line": {"type": "integer", "description": "1-based line number"},
                    "column": {
                        "type": "integer",
                        "description": "1-based column (default 1 = snaps to nearest identifier)",
                        "default": 1,
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Optional exact identifier on that line (disambiguation)",
                    },
                },
                "required": ["path", "line"],
            },
        },
    },
]

FUNCTIONS: dict[str, callable] = {
    "lsp_go_to_definition": lsp_go_to_definition,
    "lsp_find_references": lsp_find_references,
    "lsp_outline": lsp_outline,
    "lsp_workspace_symbols": lsp_workspace_symbols,
    "lsp_hover": lsp_hover,
}
