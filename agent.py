import asyncio
import datetime
import json
import logging
import re
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
        self._state = "scanning"
        self._buf   = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        if self._state == "scanning":
            m = self._OPEN_RE.search(self._buf)
            if m:
                pre         = self._buf[:m.start()]
                self._buf   = self._buf[m.end():]
                self._state = "buffering"
                return pre + self._drain()
            pm       = self._PARTIAL_OPEN_RE.search(self._buf)
            safe_end = pm.start() if pm else len(self._buf)
            out       = self._buf[:safe_end]
            self._buf = self._buf[safe_end:]
            return out
        return self._drain()

    def _drain(self) -> str:
        m = self._CLOSE_RE.search(self._buf)
        if m:
            remaining   = self._buf[m.end():].lstrip("\n")
            self._buf   = ""
            self._state = "scanning"
            return remaining
        return ""

    def finalize(self) -> str:
        if self._state == "buffering":
            out = self._buf          # unclosed block — return it so response isn't empty
        else:                        # scanning
            pm  = self._PARTIAL_OPEN_RE.search(self._buf)
            out = self._buf[:pm.start()] if pm else self._buf
        self._buf = ""
        return out

# Sync tools that do HTTP / long completions — run off the event loop.
# Memory tools call Mem0 (Groq + Gemini embeddings + Qdrant) synchronously.
_BLOCKING_SYNC_TOOLS = frozenset({
    "fetch_url",
    "web_search",
    "remember",
    "recall",
    "list_memories",
    "delete_memory",
    "shell_exec",
    "workspace_search_replace",
})


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
    return _THINK_RE.sub("", text or "").strip()


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
            kwargs["tools"]       = TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"

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
        "Emit at most one tool call per assistant message; wait for result before the next tool call.\n"
        "For code edits, use workspace_search_replace as primary. It is okay to use more sequential tool calls when each is thoughtful and based on fresh file state. Do not repeat identical failing calls; adjust snippet or strategy after each failure.\n"
    )


# Appended to system when `output_channel=telegram` (no Markdown/HTML: Telegram
# sends plain `sendMessage` with default parse mode).
_TELEGRAM_FORMAT_APPEND = (
    "\n## Output (Telegram)\n"
    "The user is on Telegram. Your visible replies are sent as plain text: "
    "no Markdown, no HTML, no `fenced` code blocks, and no LaTeX/KaTeX. "
    "Use line breaks, short lines, simple '-' bullets or numbered lines, and plain wording. "
    "For code, either very short one-line snippets or 'say file path + what to change' — "
    "not multi-line listings unless the user explicitly wants code pasted.\n"
)


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
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_content},
    ]
    turn_start = 1 + len(history)   # index of the user message; everything from here is new

    provider_used = _clients[0]["name"] if _clients else "none"
    # Pending background summarization tasks: (message_dict, asyncio.Task)
    pending_summaries: list[tuple[dict, asyncio.Task]] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        response, provider = await _call(messages)

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

        calls = _extract_calls(msg)
        if not calls:
            final = _visible_after_think(content)
            if not final and content.strip():
                repair_messages = messages + [{"role": "user", "content": _REPAIR_USER}]
                response2, provider2 = await _call(repair_messages, use_tools=False)
                provider_used = provider2
                content2 = response2.choices[0].message.content or ""
                final = _visible_after_think(content2) or content2.strip()
            if not final:
                final = "(No visible response from the model.)"
            # Store stripped visible content in history, consistent with
            # the streaming path and DESIGN §6.3 (thinking blocks stripped).
            assistant_entry["content"] = final
            log.info("done  via=%s  len=%d", provider_used, len(final))
            await _apply_finished_summaries(pending_summaries)
            return final, provider_used, messages[turn_start:], pending_summaries

        result_blocks = []
        n_tc = len(msg.tool_calls)
        for i, call in enumerate(calls):
            name = call.get("name", "")
            args = call.get("args", {})
            log.info("tool-call  %s  %s", name, str(args)[:120])
            result = await _run_tool_async(name, args)
            log.debug("run: tool_result %s → %.120s", name, result)

            nuke_summary = _extract_nuke_summary(result)
            if nuke_summary is not None:
                await _apply_finished_summaries(pending_summaries)
                return nuke_summary, provider_used, [{"role": "assistant", "content": nuke_summary, "_nuke": True}], pending_summaries

            if i < n_tc:
                tc_id = msg.tool_calls[i].id
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
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_content},
    ]
    turn_start = 1 + len(history)   # index of the user message; everything from here is new

    provider_used = _clients[0]["name"] if _clients else "none"
    # Pending background summarization tasks: (message_dict, asyncio.Task)
    pending_summaries: list[tuple[dict, asyncio.Task]] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        stream, provider = await _call_stream(messages)
        provider_used    = provider
        log.info("turn[%d]  provider=%s", iteration, provider)

        raw_parts       = []   # raw stream content including thinking tags
        visible_parts   = []   # stripped visible content for history storage
        tool_calls_acc  = {}
        stripper        = _ThinkStripper()
        tool_mode       = False   # once True, suppress text forwarding

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                tool_mode = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id or f"tc_{idx}",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = _merge_stream_fragment(
                                tool_calls_acc[idx]["name"], tc.function.name
                            )
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] = _merge_stream_fragment(
                                tool_calls_acc[idx]["arguments"], tc.function.arguments
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
        visible_content = "".join(visible_parts)

        if tool_calls_acc:
            native_tc_list = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                # Streamed argument chunks can be malformed/incomplete JSON.
                # Normalize once here so history replay stays provider-safe.
                try:
                    args_obj = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args_obj = {}
                native_tc_list.append({
                    "id":   tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(args_obj)},
                })

            messages.append({
                "role":       "assistant",
                "content":    visible_content or None,
                "tool_calls": native_tc_list,
            })

            result_messages = []
            for idx in sorted(tool_calls_acc.keys()):
                tc   = tool_calls_acc[idx]
                name = tc["name"]
                args = json.loads(next(
                    ntc["function"]["arguments"]
                    for ntc in native_tc_list if ntc["id"] == tc["id"]
                ))

                log.info("tool-call  %s  %s", name, str(args)[:120])
                yield {"type": "tool_call", "name": name, "args": args}
                result = await _run_tool_async(name, args)
                log.debug("stream: tool_result %s → %.120s", name, result)
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
                    "tool_call_id": tc["id"],
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
                yield {"type": "text_chunk", "text": rtxt}
            else:
                messages.append({"role": "assistant", "content": visible_content})
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
