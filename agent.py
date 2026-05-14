import asyncio
import datetime
import json
import logging
import re
from pathlib import Path
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
import config
from config import PROVIDERS
from summarizer import summarize_gemma
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
from tools.nuke import _NUKE_PREFIX

log = logging.getLogger(__name__)

# ── Async clients — one per provider with a key ──────────────────────────────

_clients: list[dict] = [
    {
        "name":   p["name"],
        "model":  p["model"],
        "client": AsyncOpenAI(api_key=p["api_key"], base_url=p["base_url"]),
    }
    for p in PROVIDERS if p["api_key"]
]

MAX_TOOL_ITERATIONS = 30


class DebugLogger:
    """Appends the agent's internal monologue and tool exchanges to a per-session log file.

    Files are named: `logs/debug_{session_id}.log`
    One file per session; all turns in a conversation append to the same file.
    """
    def __init__(self, session_id: str):
        self._session_id = session_id
        self._log_file = None

    def _ensure_file(self):
        if self._log_file is None and config.DEBUG_LOGGING:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self._log_file = open(log_dir / f"debug_{self._session_id}.log", "a", encoding="utf-8")

    def log(self, tag: str, content: str):
        self._ensure_file()
        if self._log_file:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            safe_content = str(content).replace("\x00", "")
            self._log_file.write(f"[{ts}] [{tag}] {safe_content}\n")
            self._log_file.flush()


# One logger per session; avoids opening a new file on every run() call.
_session_loggers: dict[str, DebugLogger] = {}


def _get_session_logger(session_id: str) -> DebugLogger:
    if session_id not in _session_loggers:
        _session_loggers[session_id] = DebugLogger(session_id)
    return _session_loggers[session_id]


# Closed reasoning blocks stripped from user-visible output (opening tag → matching close).
_THINK_RE = re.compile(
    r"<(thought|think|thinking|redacted_reasoning|redacted_thinking)[\s>].*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

_REPAIR_USER = (
    "Your previous assistant message had no user-visible text after removing "
    "closed reasoning blocks such as <thought>, <thinking>, or <redacted_reasoning> (or the reply was empty). "
    "Reply again with ONLY the answer the user should see — no reasoning tags."
)

# Tool results longer than this get summarized for history; shorter ones kept verbatim.
_HISTORY_SUMMARIZE_THRESHOLD = 8000


class _ThinkStripper:
    """Stream-safe removal of <thought/think/thinking> blocks.

    State machine: scanning → buffering → scanning (loops).
    A closing tag returns to scanning so multiple interleaved
    thinking/output/thinking cycles are all handled correctly.

    Backtick code spans and ``` fences are treated as opaque: a thinking tag
    that appears inside either context passes through verbatim and does NOT
    activate the stripper.  This is tracked via a running backtick parity
    (``_bt_parity``) over the raw stream.  Every backtick contributes,
    including those in ``` sequences — a fence opener is 3 backticks (odd),
    so it flips parity just like a single backtick, protecting its contents
    for free, and the matching closer flips it back.  Lone unmatched backticks
    inside genuine thinking blocks can theoretically desync parity, but that
    is a degenerate edge-case.
    """
    _OPEN_RE  = re.compile(r"<(thought|think|thinking)[\s>]", re.IGNORECASE)
    _CLOSE_RE = re.compile(r"</(thought|think|thinking)>",    re.IGNORECASE)
    # Any suffix of the buffer that could be the start of an opening tag.
    _PARTIAL_OPEN_RE = re.compile(
        r"(?:<thought[\s>]?|<thinking[\s>]?|<think[\s>]?|<thinkin|<thinki"
        r"|<thin|<thi|<tho(?:u(?:g(?:ht?)?)?)?|<th|<t|<)$",
        re.IGNORECASE,
    )

    def __init__(self):
        self._state     = "scanning"
        self._buf       = ""
        # Backtick parity of raw stream content already processed (emitted or
        # suppressed) BEFORE self._buf.  0 = outside a code span; 1 = inside.
        self._bt_parity = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bt_delta(text: str) -> int:
        """Net backtick parity change for text.

        Every backtick (single or triple) contributes to the count.
        A ``` fence opener/closer is 3 backticks (odd), so it flips parity
        exactly like a single backtick, protecting its contents for free.
        """
        return text.count("`") & 1

    def _parity_at(self, pos: int) -> int:
        """Code-span parity at position pos in self._buf."""
        return (self._bt_parity + self._bt_delta(self._buf[:pos])) & 1

    def _advance(self, text: str) -> None:
        """Record that text (emitted or suppressed) has been processed."""
        self._bt_parity = (self._bt_parity + self._bt_delta(text)) & 1

    # ── public interface ─────────────────────────────────────────────────────

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        if self._state == "scanning":
            search_from = 0
            while True:
                m = self._OPEN_RE.search(self._buf, search_from)
                if not m:
                    break
                if self._parity_at(m.start()) == 1:
                    # Tag is inside a backtick code span — treat as literal text.
                    search_from = m.end()
                    continue
                # Tag is outside any code span — strip the thinking block.
                pre         = self._buf[:m.start()]
                self._buf   = self._buf[m.end():]
                self._advance(pre)
                self._state = "buffering"
                tail = self._drain()
                self._advance(tail)
                return pre + tail

            # No real thinking tag found; hold back any partial tag, but only
            # when the partial match itself is outside a code span.
            pm = self._PARTIAL_OPEN_RE.search(self._buf)
            if pm and self._parity_at(pm.start()) == 0:
                safe_end = pm.start()
            else:
                safe_end = len(self._buf)
            out       = self._buf[:safe_end]
            self._buf = self._buf[safe_end:]
            self._advance(out)
            return out

        # buffering state
        tail = self._drain()
        self._advance(tail)
        return tail

    def _drain(self) -> str:
        m = self._CLOSE_RE.search(self._buf)
        if m:
            suppressed  = self._buf[:m.end()]          # content + closing tag
            remaining   = self._buf[m.end():].lstrip("\n")
            self._buf   = ""
            self._state = "scanning"
            self._advance(suppressed)                  # backticks in block affect parity
            return remaining
        return ""

    def finalize(self) -> str:
        if self._state == "buffering":
            # Unclosed thinking block: drop rather than leaking raw tags.
            out = ""
        else:
            pm = self._PARTIAL_OPEN_RE.search(self._buf)
            if pm and self._parity_at(pm.start()) == 0:
                out = self._buf[:pm.start()]
            else:
                out = self._buf
        self._buf = ""
        return out

# Sync tools that do HTTP / long completions — run off the event loop.
# Memory tools call Mem0 (Groq + Gemini embeddings + Qdrant) synchronously.
_BLOCKING_SYNC_TOOLS = frozenset({
    "fetch_url",
    "web_search",
    "youtube_transcript",
    "remember",
    "recall",
    "list_memories",
    "delete_memory",
    "shell_exec",
    "workspace_search_replace",
    "lsp_go_to_definition",
    "lsp_find_references",
    "lsp_outline",
    "lsp_workspace_symbols",
    "lsp_hover",
})

# Tools that never mutate workspace files, shell session, or destructive memory
# state — safe to execute concurrently within one assistant tool-call batch when
# the provider emits multiple tool_calls at once.
_PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset({
    "workspace_read",
    "workspace_grep",
    "web_search",
    "fetch_url",
    "youtube_transcript",
    "recall",
    "list_memories",
    "lsp_go_to_definition",
    "lsp_find_references",
    "lsp_outline",
    "lsp_workspace_symbols",
    "lsp_hover",
})


def _tool_batch_parallel_eligible(names: list[str]) -> bool:
    """True when every tool in the batch may run concurrently on the server."""
    if len(names) < 2:
        return False
    return all(n in _PARALLEL_SAFE_TOOLS for n in names)


def _tool_specs_from_message(msg) -> list[tuple[str, str, dict]]:
    """``(tool_call_id, name, args)`` in assistant ``tool_calls`` order."""
    specs: list[tuple[str, str, dict]] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        specs.append((tc.id, tc.function.name, args))
    return specs


async def _run_tool_specs_to_results(
    specs: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict, str]]:
    """Run tools in model order; parallelize read-only batches when enabled.

    Each spec is ``(tool_call_id, tool_name, args_dict)``. Returns rows
    ``(tool_call_id, tool_name, args_dict, result_str)`` in the same order.
    """
    if not specs:
        return []
    names = [s[1] for s in specs]
    if config.AGENT_PARALLEL_TOOL_CALLS and _tool_batch_parallel_eligible(names):
        results = await asyncio.gather(
            *[_run_tool_async(name, args) for _, name, args in specs]
        )
        return [
            (specs[i][0], specs[i][1], specs[i][2], results[i])
            for i in range(len(specs))
        ]
    out: list[tuple[str, str, dict, str]] = []
    for tc_id, name, args in specs:
        res = await _run_tool_async(name, args)
        out.append((tc_id, name, args, res))
    return out


_REGISTERED_TOOL_NAMES: frozenset[str] = frozenset(TOOL_FUNCTIONS)


def _merge_stream_tool_name(current: str, fragment: str) -> tuple[str, bool]:
    """Merge streamed ``function.name`` fragments.

    Returns ``(merged_name, split_new_slot)``. When ``split_new_slot`` is True,
    the caller must **not** update the current slot's name — ``merged_name`` is
    the new tool's name and belongs in a **new** slot. This handles providers
    (notably Gemini) that reuse the same ``index`` for parallel tool calls while
    streaming distinct registered tool names, which would otherwise concatenate
    into nonsense like ``workspace_readworkspace_grep``.
    """
    if not fragment:
        return current, False
    if not current:
        return fragment, False
    if fragment.startswith(current):
        return fragment, False
    if current.startswith(fragment):
        return current, False
    cur = current.strip()
    frag = fragment.strip()
    if cur in _REGISTERED_TOOL_NAMES and frag in _REGISTERED_TOOL_NAMES and cur != frag:
        return fragment, True
    return current + fragment, False


def _arguments_stream_json_complete(raw: str) -> bool:
    """True when ``raw`` parses as JSON (object or array); used to route arg deltas."""
    if raw is None or not str(raw).strip():
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def _stream_resolve_slot_for_name(slots: list[dict], tid: str, idx: int) -> dict:
    """Pick or create the slot that should receive ``function.name`` deltas."""
    if tid:
        for s in slots:
            if s["id"] == tid:
                return s
        for s in reversed(slots):
            if s["stream_index"] == idx and not s["id"]:
                s["id"] = tid
                return s
        s = {"id": tid, "stream_index": idx, "name": "", "arguments": ""}
        slots.append(s)
        return s
    for s in reversed(slots):
        if s["stream_index"] == idx:
            return s
    s = {"id": "", "stream_index": idx, "name": "", "arguments": ""}
    slots.append(s)
    return s


def _stream_slot_for_arguments(slots: list[dict], tid: str, idx: int) -> dict:
    """Pick the slot that should receive ``function.arguments`` deltas."""
    if tid:
        for s in slots:
            if s["id"] == tid:
                return s
    for s in reversed(slots):
        if s["stream_index"] != idx:
            continue
        if not _arguments_stream_json_complete(s["arguments"]):
            return s
    for s in reversed(slots):
        if s["stream_index"] == idx:
            return s
    s = {"id": tid or "", "stream_index": idx, "name": "", "arguments": ""}
    slots.append(s)
    return s


def _merge_stream_fragment(current: str, fragment: str) -> str:
    """Merge streamed name/arguments fragments without accidental duplication.

    Providers may send either:
      - true deltas (append-only fragments), or
      - cumulative snapshots (same full value repeated each chunk).

    Some OpenAI-compat streams repeat the full ``function.name`` on later chunks
    for the same index; naive concat doubles it — see ``test_stream_tool_merge``.
    """
    if not fragment:
        return current
    if not current:
        return fragment
    if fragment.startswith(current):
        return fragment
    if current.startswith(fragment):
        return current
    # Two JSON-looking fragments that are not prefix-related: some OpenAI-compat
    # streams (notably Gemini) re-send a full arguments object instead of a
    # strict suffix delta; concatenating would produce invalid JSON and later
    # json.loads → {}. Prefer the newer stream chunk.
    c_strip = current.lstrip()
    f_strip = fragment.lstrip()
    if c_strip.startswith("{") and f_strip.startswith("{"):
        if not (fragment.startswith(current) or current.startswith(fragment)):
            # Later chunk wins: disjoint JSON objects are almost always a full
            # re-send from the provider, not something to concatenate.
            return fragment
    return current + fragment


def _sanitize_message(m: dict) -> dict | None:
    """Keep only API-safe keys. Whitelist-based — any extra fields are dropped."""
    role = m.get("role")
    if role not in ("user", "assistant", "system", "tool"):
        return None
    o: dict = {"role": role}
    if m.get("content") is not None:
        o["content"] = m["content"]
    if role == "assistant" and m.get("tool_calls"):
        tcs = []
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                fn = {}
            tcs.append({
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", ""),
                },
            })
        o["tool_calls"] = tcs
    if role == "tool":
        o["tool_call_id"] = m.get("tool_call_id", "")
        o["content"] = m.get("content", "")
        if m.get("name"):
            o["name"] = m["name"]
    return o


def _sanitize_history(history: list[dict]) -> list[dict]:
    return [sm for m in history if (sm := _sanitize_message(m)) is not None]


def _visible_after_think(text: str) -> str:
    cleaned = _THINK_RE.sub("", text or "")
    # Handle malformed/unclosed reasoning tags defensively so visible output
    # never contains dangling "<thought"/"<thinking" fragments at tail.
    cleaned = re.sub(
        r"<(thought|think|thinking|redacted_reasoning|redacted_thinking)(?:\b[^>]*)?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _extract_nuke_summary(tool_result: str) -> str | None:
    if not isinstance(tool_result, str) or not tool_result.startswith(_NUKE_PREFIX):
        return None
    payload = tool_result[len(_NUKE_PREFIX):]
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    summary = str(obj.get("summary", "")).strip()
    return summary or None


def _user_content_as_text(user_content: "str | list") -> str:
    """Extract a plain-text representation from a user content value.

    Used when a summarizer or log call needs a string even if the original
    user turn included multimodal parts (images, file chunks).
    """
    if isinstance(user_content, str):
        return user_content
    return " ".join(p["text"] for p in user_content if isinstance(p, dict) and p.get("type") == "text")




def _extract_calls(msg) -> list[dict]:
    if not getattr(msg, "tool_calls", None):
        return []
    out = []
    for tc in msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        out.append({"name": tc.function.name, "args": args})
    return out


# ── Provider calls ────────────────────────────────────────────────────────────

async def _call(messages: list[dict], use_tools: bool = True) -> tuple:
    """Non-streaming call; walks provider chain with retries."""
    if not _clients:
        raise RuntimeError("No providers configured — set at least one API key.")

    last_err = None
    for entry in _clients:
        kwargs = dict(model=entry["model"], messages=messages, max_tokens=8192)
        if use_tools:
            kwargs["tools"] = TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"
            if config.AGENT_PARALLEL_TOOL_CALLS:
                kwargs["parallel_tool_calls"] = True

        for attempt in range(3):
            try:
                resp = await entry["client"].chat.completions.create(**kwargs)
                return resp, entry["name"]
            except (RateLimitError, APIConnectionError) as e:
                last_err = e
                log.warning("provider=%s attempt=%d retryable error: %s", entry["name"], attempt, e)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except APIError as e:
                last_err = e
                log.warning("provider=%s attempt=%d api error (skipping): %s", entry["name"], attempt, e)
                break
            except Exception as e:
                last_err = e
                log.warning("provider=%s attempt=%d unexpected error (skipping): %s", entry["name"], attempt, e)
                break

    raise RuntimeError(f"All providers exhausted. Last error: {last_err}")


async def _call_stream(messages: list[dict]) -> tuple:
    """Streaming call; walks provider chain with retries."""
    if not _clients:
        raise RuntimeError("No providers configured.")

    last_err = None
    for entry in _clients:
        kwargs = dict(
            model       = entry["model"],
            messages    = messages,
            max_tokens  = 8192,
            stream      = True,
            tools       = TOOL_SCHEMAS,
            tool_choice = "auto",
        )
        if config.AGENT_PARALLEL_TOOL_CALLS:
            kwargs["parallel_tool_calls"] = True

        for attempt in range(3):
            try:
                stream = await entry["client"].chat.completions.create(**kwargs)
                return stream, entry["name"]
            except (RateLimitError, APIConnectionError) as e:
                last_err = e
                log.warning("stream provider=%s attempt=%d retryable error: %s", entry["name"], attempt, e)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except APIError as e:
                last_err = e
                log.warning("stream provider=%s attempt=%d api error (skipping): %s", entry["name"], attempt, e)
                break
            except Exception as e:
                last_err = e
                log.warning("stream provider=%s attempt=%d unexpected error (skipping): %s", entry["name"], attempt, e)
                break

    raise RuntimeError(f"All providers exhausted for streaming. Last error: {last_err}")


# ── System prompt helpers ─────────────────────────────────────────────────────

def _tool_docs() -> str:
    return (
        "## Tool usage\n"
        "Use native API function/tool_calls only (no XML or fenced tool syntax).\n"
        "You may emit multiple tool_calls in one assistant message only when every call is "
        "read-only discovery (e.g. several workspace_read / workspace_grep / lsp_outline / "
        "lsp_go_to_definition / lsp_find_references / lsp_workspace_symbols / lsp_hover / web_search / fetch_url / "
        "youtube_transcript / recall / list_memories together). Prefer that batch when the calls are "
        "independent (none needs another tool's return to choose its arguments); if you must read A "
        "before you can decide B's args, call A first, then batch the rest. For writes, shell session, or "
        "memory mutations, emit one tool call per message and wait for its result before the next.\n"
        "For Python symbols prefer lsp_*; for plain-text or non-Python discovery use workspace_grep.\n"
        "Never emit empty arguments: every required field must be present in the tool JSON; prose does not substitute for arguments. After missing-argument errors, fix fields—do not retry `{}` or the same empty pattern.\n"
        "For code edits, use workspace_search_replace as primary. Keep replacements narrow and surgical: use short, unique old_string snippets around only the target lines. Avoid whole-file old/new payloads unless the user explicitly asks for a full rewrite. It is okay to use multiple sequential calls when each call is thoughtful and based on fresh file state. Do not repeat identical failing calls; after a failure, change snippet or strategy.\n"
    )


# Appended to system when `output_channel=telegram`.
_TELEGRAM_FORMAT_APPEND = (
    "\n## Output (Telegram)\n"
    "The user is on Telegram. Keep visible replies plain and compact: no Markdown emphasis, "
    "no double-asterisk bold (`**text**`), no HTML tags, no fenced code blocks, and no LaTeX/KaTeX. "
    "Use simple lines/bullets and direct wording. "
    "For code, prefer very short one-line snippets or describe file path + what to change unless the user explicitly asks to paste code.\n"
)


async def _prefetch_memories(user_text: str, history: list[dict]) -> list[str]:
    """Query the memory store before the first LLM call and return relevant strings.

    Query = user message + last assistant snippet (for cross-domain coverage,
    e.g. "algorithms homework" paired with a prior math/LaTeX turn pulls up
    formatting preferences).  Returns [] on any failure so the turn is never
    blocked.
    """
    from tools.memory import recall_prefetch

    query_parts = [user_text.strip()]
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            txt = msg.get("content") or ""
            if isinstance(txt, list):
                txt = " ".join(b.get("text", "") for b in txt if isinstance(b, dict))
            snippet = txt.strip()[:200]
            if snippet:
                query_parts.append(snippet)
            break
    query = "\n".join(query_parts)

    try:
        return await asyncio.to_thread(
            recall_prefetch,
            query,
            top_k=config.MEMORY_PREFETCH_TOP_K,
            threshold=config.MEMORY_PREFETCH_THRESHOLD,
        )
    except Exception as exc:
        log.warning("memory prefetch failed (skipping): %s", exc)
        return []


def _memories_user_block(memories: list[str]) -> dict:
    """Format prefetched memories as a user-role message immediately before the real user message.

    Injected at the end of context (highest attention position) rather than
    after the system prompt (lowest marginal attention when history is long).
    """
    body = "\n".join(f"- {m}" for m in memories)
    return {
        "role": "user",
        "content": (
            "[Memory — these may help; call recall for more; call remember for new facts]\n"
            + body
        ),
    }


def _build_system_prompt(*, output_channel: str = "default") -> str:
    # Inject today's date so the model has a concrete present to reason
    # against its January 2025 training cutoff. Without this anchor,
    # "post-cutoff" stays abstract and the model's RLHF-trained reflex to
    # disclaim recent info as possible hallucination tends to fire even on
    # tool-grounded data. See DESIGN §10.
    today = datetime.date.today().strftime("%A %Y-%m-%d")
    base = config.SYSTEM_PROMPT_PATH.read_text()
    out = f"Today's date: {today}\n\n{base}\n\n{_tool_docs()}"
    if output_channel == "telegram":
        out += _TELEGRAM_FORMAT_APPEND
    return out


# ── Tool execution helper ─────────────────────────────────────────────────────

async def _run_tool_async(name: str, args: dict) -> str:
    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"
    fn = TOOL_FUNCTIONS[name]

    def _invoke_sync() -> str:
        try:
            return str(fn(**args))
        except Exception as e:
            return f"Error in {name}: {e}"

    if name in _BLOCKING_SYNC_TOOLS:
        return await asyncio.to_thread(_invoke_sync)
    return _invoke_sync()


_PAGINATION_RE = re.compile(
    r'\[…[^\]]*call fetch_url with offset=(\d+)[^\]]*\]'
)


_HISTORY_SUMMARY_SYSTEM = (
    "You compress large tool outputs for conversation history storage. "
    "Reply with substantive takeaways from the TOOL OUTPUT section only: key facts, numbers, "
    "names, dates, errors, and actionable conclusions. Short bullets or tight prose — no preamble. "
    "Do NOT restate or summarize the user's request, tool name, or tool arguments in your reply "
    "(the reader already has that from the message). Do NOT write meta lines like 'The user asked…' "
    "or 'Given this tool call…'. Do NOT narrate what anyone expected or wanted (e.g. 'looking for "
    "albums') — that is not data. DO report factual outcomes that appear in the tool output: "
    "successful extractions, empty or missing sections, no matches, not found, partial results, "
    "and explicit errors — all grounded in the output text itself."
)


async def _summarize_for_history(name: str, args: dict, user_message: str, content: str) -> str:
    """Summarize a large tool result for storage in conversation history.

    Uses Gemma 26B (same model the fetch tool uses) — ~10x faster than the
    primary and fine-grained enough for compact structured summaries. This
    keeps `_clients` (the primary provider chain) reserved for the agent loop.

    Falls back to truncating the raw content to `_HISTORY_SUMMARIZE_THRESHOLD`
    chars if the summarizer fails or returns nothing visible.
    """
    # For fetch_url, preserve pagination note — the summarizer would drop it,
    # but the agent needs it to know the next offset to request.
    pagination_note = None
    if name == "fetch_url":
        m = _PAGINATION_RE.search(content)
        if m:
            pagination_note = m.group(0)

    # Context blocks are for the summarizer's reasoning only; _HISTORY_SUMMARY_SYSTEM
    # forbids echoing them in the model's reply.
    user_prompt = (
        "Use the following only to decide what matters in the tool output. "
        "Do not repeat them in your reply.\n\n"
        f"User request: {user_message}\n"
        f"Tool: {name}\n"
        f"Args: {json.dumps(args)}\n\n"
        "---\n\n"
        f"TOOL OUTPUT:\n{content}"
    )

    try:
        summary = await asyncio.to_thread(
            summarize_gemma, _HISTORY_SUMMARY_SYSTEM, user_prompt
        )
        summary = _visible_after_think(summary)
        if summary:
            if pagination_note:
                summary = f"{summary}\n\n{pagination_note}"
            log.info("history-summary  %s  %d→%d chars", name, len(content), len(summary))
            return (
                f"[history summary of {name}]\n"
                "This tool response was summarized for context efficiency. Takeaways:\n"
                f"{summary}"
            )
    except Exception as e:
        log.warning("history-summary failed  %s: %s", name, e)
    return content[:_HISTORY_SUMMARIZE_THRESHOLD]


# ── Non-blocking summary application ─────────────────────────────────────────

async def _apply_finished_summaries(
    pending: list[tuple[dict, asyncio.Task]]
) -> None:
    """Opportunistically apply any summary tasks that have already completed.

    Called only when **no further LLM tool rounds** will run in this user turn
    (final text path, max-iter exit, or stream `done`), or from `main.py` after
    persist — never between tool rounds — so every `_call` within the turn still
    sees full prior tool outputs (DESIGN §4.1).

    A slow summarizer MUST NOT stall a turn: tasks still in flight stay in
    `pending` for the caller to drain post-persist. A single `asyncio.sleep(0)`
    tick lets fast summarizers apply before return when this runs at turn end.
    """
    if not pending:
        return
    await asyncio.sleep(0)
    remaining: list[tuple[dict, asyncio.Task]] = []
    for msg_dict, task in pending:
        if task.done():
            try:
                result = task.result()
                if isinstance(result, str):
                    msg_dict["content"] = result
            except Exception as e:
                log.warning("pending summary raised: %s", e)
        else:
            remaining.append((msg_dict, task))
    pending[:] = remaining


# ── Non-streaming run (used by /chat endpoint) ────────────────────────────────

async def run(
    user_content: "str | list",
    history: list[dict],
    *,
    session_id: str = "default",
    output_channel: str = "default",
) -> tuple[str, str, list[dict], list[tuple[dict, asyncio.Task]]]:
    """Returns (final_response, provider, turn_messages, pending_summaries).

    `user_content` is either a plain string or a multimodal content list
    (OpenAI vision format: [{type: "text", text: ...}, {type: "image_url", ...}, ...]).

    `turn_messages` is the full slice of messages added this turn — starting
    from the user message through to the final assistant reply, including all
    intermediate tool-call and tool-result messages. Store this in history so
    the model sees its own tool usage on the next turn.

    `pending_summaries` is a list of (message_dict, asyncio.Task) pairs for
    tool-result summaries that had not finished by the time the turn returned.
    The caller may await the tasks after persisting the turn and then update
    the stored row with the summarized content — see `main.py`. Under the
    non-blocking contract, callers MUST NOT await these in a path that stalls
    the user's response.
    """
    history = _sanitize_history(history)
    _user_text = _user_content_as_text(user_content)
    system_prompt = _build_system_prompt(output_channel=output_channel)
    prefetched = await _prefetch_memories(_user_text, history)
    mem_block = [_memories_user_block(prefetched)] if prefetched else []
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        *mem_block,
        {"role": "user", "content": user_content},
    ]
    
    session_log = _get_session_logger(session_id)
    session_log.log("USER", _user_text)
    # turn_start excludes the memory block — it's ephemeral context, not persisted history.
    turn_start = 1 + len(history) + len(mem_block)

    provider_used = _clients[0]["name"] if _clients else "none"
    # Pending background summarization tasks: (message_dict, asyncio.Task)
    pending_summaries: list[tuple[dict, asyncio.Task]] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        response, provider = await _call(messages)
        session_log.log("THOUGHT", response.choices[0].message.content or "")

        provider_used = provider
        msg     = response.choices[0].message
        content = msg.content or ""

        assistant_entry: dict = {"role": "assistant", "content": content}
        if getattr(msg, "tool_calls", None):
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if not getattr(msg, "tool_calls", None):
            final = _visible_after_think(content)
            if not final and content.strip():
                repair_messages = messages + [{"role": "user", "content": _REPAIR_USER}]
                response2, provider2 = await _call(repair_messages, use_tools=False)
                provider_used = provider2
                content2 = response2.choices[0].message.content or ""
                # Never fall back to raw repair text; malformed/unclosed thought
                # tags in repair output must not leak to users.
                final = _visible_after_think(content2)
            if not final:
                final = "(No visible response from the model.)"
            session_log.log("OUTPUT", final)
            # Store stripped visible content in history, consistent with
            # the streaming path and DESIGN §6.3 (thinking blocks stripped).
            assistant_entry["content"] = final
            log.info("done  via=%s  len=%d", provider_used, len(final))
            await _apply_finished_summaries(pending_summaries)
            return final, provider_used, messages[turn_start:], pending_summaries

        specs = _tool_specs_from_message(msg)
        rows = await _run_tool_specs_to_results(specs)
        result_blocks: list[dict] = []
        for tc_id, name, args, result in rows:
            log.info("tool-call  %s  %s", name, str(args)[:120])
            session_log.log("TOOL_CALL", f"{name}({json.dumps(args)})")
            log.debug("run: tool_result %s → %.120s", name, result)
            session_log.log("TOOL_RESULT", f"{name} -> {result}")

            nuke_summary = _extract_nuke_summary(result)
            if nuke_summary is not None:
                await _apply_finished_summaries(pending_summaries)
                return nuke_summary, provider_used, [{"role": "assistant", "content": nuke_summary, "_nuke": True}], pending_summaries

            msg_dict = {"role": "tool", "name": name, "tool_call_id": tc_id, "content": result}
            result_blocks.append(msg_dict)
            if len(result) > _HISTORY_SUMMARIZE_THRESHOLD:
                task = asyncio.create_task(_summarize_for_history(name, args, _user_text, result))
                pending_summaries.append((msg_dict, task))

        messages.extend(result_blocks)

    await _apply_finished_summaries(pending_summaries)
    return "Reached max tool iterations.", provider_used, messages[turn_start:], pending_summaries


# ── Streaming run (used by /chat/stream endpoint) ─────────────────────────────

async def run_stream(
    user_content: "str | list",
    history: list[dict],
    *,
    session_id: str = "default",
    output_channel: str = "default",
):
    """
    Async generator yielding event dicts:
      {"type": "tool_call",   "name": str, "args": dict}
      {"type": "tool_result", "name": str, "result": str}
      {"type": "text_chunk",  "text": str}
      {"type": "done",        "provider": str, "turn_messages": list, "pending_summaries": list}
      {"type": "error",       "detail": str}

    `user_content` is either a plain string or a multimodal content list
    (OpenAI vision format). See `run()` for full contract.

    The `done` event's `pending_summaries` is a list of (message_dict,
    asyncio.Task) pairs for tool-result summaries not yet finished. The caller
    (main.py) drains them in the background after persisting the turn — see
    DESIGN §6.5. `pending_summaries` MUST be popped by the caller before the
    event is JSON-serialized onto the SSE stream.
    """
    history = _sanitize_history(history)
    _user_text = _user_content_as_text(user_content)
    system_prompt = _build_system_prompt(output_channel=output_channel)
    prefetched = await _prefetch_memories(_user_text, history)
    mem_block = [_memories_user_block(prefetched)] if prefetched else []
    if prefetched:
        yield {"type": "memory_prefetch", "count": len(prefetched), "memories": prefetched}
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        *mem_block,
        {"role": "user", "content": user_content},
    ]
    session_log = _get_session_logger(session_id)
    session_log.log("USER", _user_text)
    # turn_start excludes the memory block — it's ephemeral context, not persisted history.
    turn_start = 1 + len(history) + len(mem_block)

    provider_used = _clients[0]["name"] if _clients else "none"
    # Pending background summarization tasks: (message_dict, asyncio.Task)
    pending_summaries: list[tuple[dict, asyncio.Task]] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        stream, provider = await _call_stream(messages)
        provider_used    = provider
        log.info("turn[%d]  provider=%s", iteration, provider)
        
        # The raw stream contains the thoughts; we capture them for the debug log.
        raw_parts       = []   # raw stream content including thinking tags
        visible_parts   = []   # stripped visible content for history storage
        tool_calls_slots: list[dict] = []
        stripper        = _ThinkStripper()
        tool_mode       = False   # once True, suppress text forwarding

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                tool_mode = True
                for tc in delta.tool_calls:
                    if not tc.function:
                        continue
                    idx = tc.index if tc.index is not None else 0
                    tid = (tc.id or "").strip()
                    fn = tc.function
                    name_frag = (fn.name or "").strip()
                    has_args = fn.arguments is not None

                    slot: dict | None = None
                    if name_frag:
                        slot = _stream_resolve_slot_for_name(tool_calls_slots, tid, idx)
                        merged, split = _merge_stream_tool_name(slot["name"], name_frag)
                        if split:
                            new_slot = {
                                "id": tid or "",
                                "stream_index": idx,
                                "name": name_frag,
                                "arguments": "",
                            }
                            tool_calls_slots.append(new_slot)
                            slot = new_slot
                        else:
                            slot["name"] = merged
                        if tid and not slot["id"]:
                            slot["id"] = tid
                    if has_args:
                        if slot is None:
                            slot = _stream_slot_for_arguments(tool_calls_slots, tid, idx)
                        if tid and not slot["id"]:
                            slot["id"] = tid
                        slot["arguments"] = _merge_stream_fragment(
                            slot["arguments"], fn.arguments
                        )

            if delta.content and not tool_mode:
                raw_parts.append(delta.content)
                prev_state = stripper._state
                forwarded  = stripper.feed(delta.content)
                # Thinking block just closed → tell the UI to drop the stale count.
                # The model may still be generating before streaming its first output
                # token, so we keep the dots visible rather than hiding entirely.
                if prev_state == "buffering" and stripper._state == "scanning":
                    yield {"type": "thinking_done"}
                if forwarded:
                    visible_parts.append(forwarded)
                    yield {"type": "text_chunk", "text": forwarded}
                elif stripper._state == "buffering":
                    yield {"type": "thinking_chars", "count": len(stripper._buf)}

        # Flush any partial thought buffer
        tail = stripper.finalize()
        if tail and not tool_mode:
            visible_parts.append(tail)
            yield {"type": "text_chunk", "text": tail}
            raw_parts.append(tail)

        full_content    = "".join(raw_parts)
        session_log.log("THOUGHT", full_content)
        visible_content = "".join(visible_parts)

        if tool_calls_slots:
            native_tc_list = []
            for i, tc in enumerate(tool_calls_slots):
                tc_id = tc["id"] or f"tc_{i}"
                # Streamed argument chunks can be malformed/incomplete JSON.
                # Normalize once here so history replay stays provider-safe.
                try:
                    args_obj = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args_obj = {}
                native_tc_list.append({
                    "id":   tc_id,
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(args_obj)},
                })

            messages.append({
                "role":       "assistant",
                "content":    visible_content or None,
                "tool_calls": native_tc_list,
            })

            specs: list[tuple[str, str, dict]] = []
            for ntc in native_tc_list:
                try:
                    args_obj = json.loads(ntc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args_obj = {}
                specs.append((ntc["id"], ntc["function"]["name"], args_obj))

            for tc_id, name, args in specs:
                log.info("tool-call  %s  %s", name, str(args)[:120])
                session_log.log("TOOL_CALL", f"{name}({json.dumps(args)})")
                yield {"type": "tool_call", "name": name, "args": args}

            rows = await _run_tool_specs_to_results(specs)
            result_messages = []
            for tc_id, name, args, result in rows:
                log.debug("stream: tool_result %s → %.120s", name, result)
                session_log.log("TOOL_RESULT", f"{name} -> {result}")
                yield {"type": "tool_result", "name": name, "result": result}

                nuke_summary = _extract_nuke_summary(result)
                if nuke_summary is not None:
                    await _apply_finished_summaries(pending_summaries)
                    yield {"type": "text_chunk", "text": nuke_summary}
                    yield {
                        "type": "done",
                        "provider": provider_used,
                        "turn_messages": [{"role": "assistant", "content": nuke_summary, "_nuke": True}],
                        "pending_summaries": pending_summaries,
                    }
                    return

                msg_dict = {
                    "role":         "tool",
                    "name":         name,
                    "tool_call_id": tc_id,
                    "content":      result,
                }
                result_messages.append(msg_dict)
                if len(result) > _HISTORY_SUMMARIZE_THRESHOLD:
                    task = asyncio.create_task(_summarize_for_history(name, args, _user_text, result))
                    pending_summaries.append((msg_dict, task))

            messages.extend(result_messages)

        else:
            # Text was already forwarded chunk-by-chunk during streaming. The
            # repair gate keys on VISIBLE content, not raw — a stream that is
            # entirely <thinking>...</thinking> has non-empty raw content but
            # zero visible content, and the user still saw nothing. §4.3.
            if not visible_content.strip():
                if not full_content.strip():
                    yield {"type": "error", "detail": "Empty response from model."}
                    return
                # Thinking-only reply → repair. Run the repair against a scratch
                # message list so the scaffold (`_REPAIR_USER` + the empty-visible
                # assistant placeholder) does NOT leak into stored history.
                repair_messages = messages + [
                    {"role": "assistant", "content": full_content},
                    {"role": "user", "content": _REPAIR_USER},
                ]
                response2, provider2 = await _call(repair_messages, use_tools=False)
                provider_used = provider2
                rtxt = _visible_after_think(response2.choices[0].message.content or "")
                if not rtxt:
                    yield {"type": "error", "detail": "Empty response after repair."}
                    return
                messages.append({"role": "assistant", "content": rtxt})
                session_log.log("OUTPUT", rtxt)
                yield {"type": "text_chunk", "text": rtxt}
            else:
                messages.append({"role": "assistant", "content": visible_content})
                session_log.log("OUTPUT", visible_content)
            await _apply_finished_summaries(pending_summaries)
            yield {
                "type": "done",
                "provider": provider_used,
                "turn_messages": messages[turn_start:],
                "pending_summaries": pending_summaries,
            }
            return

    await _apply_finished_summaries(pending_summaries)
    yield {"type": "error", "detail": "Reached max tool iterations."}
