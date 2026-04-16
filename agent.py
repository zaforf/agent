import asyncio
import json
import logging
import re
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
from config import PROVIDERS
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
from tools.self_modify import get_system_prompt

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

MAX_TOOL_ITERATIONS = 10

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
    State: scanning → buffering → passthrough.
    Once past any thought block, all chunks are forwarded immediately.
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
        if self._state == "passthrough":
            return chunk
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
            self._state = "passthrough"
            return remaining
        return ""

    def finalize(self) -> str:
        if self._state == "buffering":
            out = self._buf          # unclosed block — return it so response isn't empty
        elif self._state == "scanning":
            pm  = self._PARTIAL_OPEN_RE.search(self._buf)
            out = self._buf[:pm.start()] if pm else self._buf
        else:
            out = self._buf
        self._buf = ""
        return out

# Sync tools that do HTTP / long completions — run off the event loop.
_BLOCKING_SYNC_TOOLS = frozenset({"fetch_url"})


def _sanitize_message(m: dict) -> dict | None:
    """Keep only API-safe keys. Drops UI-only fields like `steps`."""
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
    return o


def _sanitize_history(history: list[dict]) -> list[dict]:
    return [sm for m in history if (sm := _sanitize_message(m)) is not None]


def _visible_after_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()




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
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except APIError as e:
                last_err = e
                break
            except Exception as e:
                last_err = e
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
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except APIError as e:
                last_err = e
                break
            except Exception as e:
                last_err = e
                break

    raise RuntimeError(f"All providers exhausted for streaming. Last error: {last_err}")


# ── System prompt helpers ─────────────────────────────────────────────────────

def _tool_docs() -> str:
    lines = [
        "## Tools\n",
        "Use the API **function / tool_calls** mechanism only (no XML or fenced code for tools).\n",
    ]
    for schema in TOOL_SCHEMAS:
        fn       = schema["function"]
        params   = fn.get("parameters", {}).get("properties", {})
        param_str = ", ".join(f"{k}: {v.get('type','string')}" for k, v in params.items())
        lines.append(f"**{fn['name']}**({param_str}): {fn['description']}\n")
    return "\n".join(lines)


def _build_system_prompt() -> str:
    return f"{get_system_prompt()}\n\n{_tool_docs()}"


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


async def _summarize_for_history(name: str, args: dict, user_message: str, content: str) -> str:
    """Summarize a large tool result for storage in conversation history."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise summarizer. Given a tool call and its output, produce a compact "
                "summary that retains everything relevant to the user's request and the tool's purpose. "
                "Keep key facts, numbers, decisions, errors, and conclusions. Omit boilerplate and repetition."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User request: {user_message}\n"
                f"Tool called: {name}\n"
                f"Tool args: {json.dumps(args)}\n\n"
                f"Tool output:\n{content}"
            ),
        },
    ]
    try:
        resp, _ = await _call(messages, use_tools=False)
        summary = _visible_after_think(resp.choices[0].message.content or "")
        if summary:
            log.info("history-summary  %s  %d→%d chars", name, len(content), len(summary))
            return f"[history summary of {name}]\n{summary}"
    except Exception as e:
        log.warning("history-summary failed  %s: %s", name, e)
    return content[:_HISTORY_SUMMARIZE_THRESHOLD]


# ── Non-streaming run (used by /chat endpoint) ────────────────────────────────

async def run(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Returns (final_response, steps)."""
    steps: list[dict] = []
    history = _sanitize_history(history)
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        *history,
        {"role": "user", "content": user_message},
    ]

    provider_used = _clients[0]["name"] if _clients else "none"
    # Pending background summarization tasks: (message_dict, asyncio.Task)
    pending_summaries: list[tuple[dict, asyncio.Task]] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        # Resolve any background summaries from the previous iteration before
        # the next LLM call, so history is compact going forward.
        if pending_summaries:
            results = await asyncio.gather(*(t for _, t in pending_summaries), return_exceptions=True)
            for (msg_dict, _), summary in zip(pending_summaries, results):
                if isinstance(summary, str):
                    msg_dict["content"] = summary
            pending_summaries.clear()

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
                messages.append({"role": "user", "content": _REPAIR_USER})
                response2, provider2 = await _call(messages, use_tools=False)
                provider_used = provider2
                content2 = response2.choices[0].message.content or ""
                final = _visible_after_think(content2) or content2.strip()
            if not final:
                final = "(No visible response from the model.)"
            log.info("done  via=%s  len=%d", provider_used, len(final))
            if _clients and provider_used != _clients[0]["name"]:
                final = f"[{provider_used} fallback]\n{final}"
            return final, steps

        result_blocks = []
        n_tc = len(msg.tool_calls)
        for i, call in enumerate(calls):
            name = call.get("name", "")
            args = call.get("args", {})
            log.info("tool-call  %s  %s", name, str(args)[:120])
            steps.append({"type": "tool_call",   "name": name, "args": args})
            result = await _run_tool_async(name, args)
            log.debug("run: tool_result %s → %.120s", name, result)
            steps.append({"type": "tool_result", "name": name, "result": result})

            if i < n_tc:
                tc_id = msg.tool_calls[i].id
                msg_dict = {"role": "tool", "tool_call_id": tc_id, "content": result}
                result_blocks.append(msg_dict)
                if len(result) > _HISTORY_SUMMARIZE_THRESHOLD:
                    task = asyncio.create_task(_summarize_for_history(name, args, user_message, result))
                    pending_summaries.append((msg_dict, task))

        messages.extend(result_blocks)

    return "Reached max tool iterations.", steps


# ── Streaming run (used by /chat/stream endpoint) ─────────────────────────────

async def run_stream(user_message: str, history: list[dict]):
    """
    Async generator yielding event dicts:
      {"type": "tool_call",   "name": str, "args": dict}
      {"type": "tool_result", "name": str, "result": str}
      {"type": "text_chunk",  "text": str}
      {"type": "done",        "provider": str}
      {"type": "error",       "detail": str}
    """
    history = _sanitize_history(history)
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        *history,
        {"role": "user", "content": user_message},
    ]

    provider_used = _clients[0]["name"] if _clients else "none"
    # Pending background summarization tasks: (message_dict, asyncio.Task)
    pending_summaries: list[tuple[dict, asyncio.Task]] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        # Resolve summaries from the previous iteration before calling the LLM again.
        # They run concurrently with the streaming response above, so by the time
        # we loop back here they're usually already done.
        if pending_summaries:
            results = await asyncio.gather(*(t for _, t in pending_summaries), return_exceptions=True)
            for (msg_dict, _), summary in zip(pending_summaries, results):
                if isinstance(summary, str):
                    msg_dict["content"] = summary
            pending_summaries.clear()

        stream, provider = await _call_stream(messages)
        provider_used    = provider
        log.info("turn[%d]  provider=%s", iteration, provider)

        content_parts   = []
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
                            tool_calls_acc[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

            if delta.content and not tool_mode:
                content_parts.append(delta.content)
                forwarded = stripper.feed(delta.content)
                if forwarded:
                    yield {"type": "text_chunk", "text": forwarded}

        # Flush any partial thought buffer
        tail = stripper.finalize()
        if tail and not tool_mode:
            yield {"type": "text_chunk", "text": tail}
            content_parts.append(tail)

        full_content = "".join(content_parts)

        if tool_calls_acc:
            native_tc_list = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                native_tc_list.append({
                    "id":   tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })

            messages.append({
                "role":       "assistant",
                "content":    full_content or None,
                "tool_calls": native_tc_list,
            })

            result_messages = []
            for idx in sorted(tool_calls_acc.keys()):
                tc   = tool_calls_acc[idx]
                name = tc["name"]
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                log.info("tool-call  %s  %s", name, str(args)[:120])
                yield {"type": "tool_call", "name": name, "args": args}
                result = await _run_tool_async(name, args)
                log.debug("stream: tool_result %s → %.120s", name, result)
                yield {"type": "tool_result", "name": name, "result": result}

                msg_dict = {
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result,
                }
                result_messages.append(msg_dict)
                if len(result) > _HISTORY_SUMMARIZE_THRESHOLD:
                    task = asyncio.create_task(_summarize_for_history(name, args, user_message, result))
                    pending_summaries.append((msg_dict, task))

            messages.extend(result_messages)

        else:
            # Text was already forwarded chunk-by-chunk during streaming.
            # If nothing was yielded (model replied entirely inside thought tags),
            # make a repair call.
            if not full_content.strip():
                yield {"type": "error", "detail": "Empty response from model."}
                return
            if not any(content_parts):
                messages.append({"role": "assistant", "content": full_content})
                messages.append({"role": "user", "content": _REPAIR_USER})
                response2, provider2 = await _call(messages, use_tools=False)
                provider_used = provider2
                rtxt = _visible_after_think(response2.choices[0].message.content or "")
                if not rtxt:
                    yield {"type": "error", "detail": "Empty response after repair."}
                    return
                yield {"type": "text_chunk", "text": rtxt}
            yield {"type": "done", "provider": provider_used}
            return

    yield {"type": "error", "detail": "Reached max tool iterations."}
