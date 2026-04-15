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

# Explicit <tool_call>…</tool_call> text fallback (XML only — no Python syntax)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Thought/reasoning blocks to strip from final output
_THINK_RE     = re.compile(r"<(thought|think|thinking)>.*?</(thought|think|thinking)>", re.DOTALL)


# ── Streaming thought-stripper ────────────────────────────────────────────────

class _ThinkStripper:
    """
    Streaming-safe removal of <thought/think/thinking> blocks.
    Gemma 4 emits these at the start of a response before actual content.

    States:
      scanning    – looking for an opening tag
      buffering   – inside a thought block; accumulate, don't forward
      passthrough – past any thought block; forward everything immediately

    Key correctness property: _PARTIAL_OPEN_RE detects when the end of the
    buffer could be the beginning of an opening tag that is split across chunk
    boundaries. We never flush those trailing bytes until we know whether they
    complete a tag or not. This prevents partial tags like "<thought" from
    leaking into the output.
    """
    _OPEN_RE  = re.compile(r"<(thought|think|thinking)>", re.IGNORECASE)
    _CLOSE_RE = re.compile(r"</(thought|think|thinking)>", re.IGNORECASE)

    # Matches any suffix of the buffer that is a valid PREFIX of one of the
    # three opening tags. Used to hold back bytes that might complete a tag
    # in the next chunk. Ends with $ so re.search finds the rightmost such
    # suffix (the one touching the end of the string).
    _PARTIAL_OPEN_RE = re.compile(
        r"<(t(h(o(u(g(h(t>?)?)?)?)?)?|i(n(k(>|(i(n(g>?)?)?)?)?)?)?)?)?$",
        re.IGNORECASE,
    )

    def __init__(self):
        self._state = "scanning"
        self._buf   = ""

    def feed(self, chunk: str) -> str:
        """Returns text safe to forward to the client."""
        if self._state == "passthrough":
            return chunk

        self._buf += chunk

        if self._state == "scanning":
            # Check for a complete opening tag anywhere in the accumulated buffer
            m = self._OPEN_RE.search(self._buf)
            if m:
                pre         = self._buf[:m.start()]
                self._buf   = self._buf[m.end():]
                self._state = "buffering"
                log.debug("thought-stripper: opening tag found, buffering")
                return pre + self._drain()

            # No complete tag yet. Find the largest safe prefix: everything
            # before any potential partial tag at the buffer's end.
            pm       = self._PARTIAL_OPEN_RE.search(self._buf)
            safe_end = pm.start() if pm else len(self._buf)
            out       = self._buf[:safe_end]
            self._buf = self._buf[safe_end:]
            return out

        # state == "buffering"
        return self._drain()

    def _drain(self) -> str:
        m = self._CLOSE_RE.search(self._buf)
        if m:
            thought_chars = m.start()
            log.debug("thought-stripper: closing tag found, stripped %d chars", thought_chars)
            remaining   = self._buf[m.end():].lstrip("\n")
            self._buf   = ""
            self._state = "passthrough"
            return remaining
        return ""

    def finalize(self) -> str:
        """Call after stream ends; returns any buffered non-thought text."""
        if self._state == "buffering":
            # Thought block was never closed — discard entirely
            log.debug("thought-stripper: stream ended inside thought block, discarding")
            out = ""
        elif self._state == "scanning":
            # Discard any partial tag candidate at the end of the buffer
            pm  = self._PARTIAL_OPEN_RE.search(self._buf)
            out = self._buf[:pm.start()] if pm else self._buf
        else:
            out = self._buf
        self._buf = ""
        return out


# ── Tool call parsing (XML fallback only) ────────────────────────────────────

def _parse_tool_calls(content: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(content):
        try:
            calls.append(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            pass
    return calls


def _extract_calls(msg) -> list[dict]:
    if getattr(msg, "tool_calls", None):
        out = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out.append({"name": tc.function.name, "args": args})
        return out
    return _parse_tool_calls(msg.content or "")


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
        "Preferred call format (use this if structured tool calls aren't available):\n",
        "```\n<tool_call>{\"name\": \"tool_name\", \"args\": {\"param\": \"value\"}}</tool_call>\n```\n",
        "You will receive `<tool_result>` messages. Give your final response only after all tool calls resolve.\n",
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

def _run_tool(name: str, args: dict) -> str:
    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"
    try:
        return str(TOOL_FUNCTIONS[name](**args))
    except Exception as e:
        return f"Error in {name}: {e}"


# ── Non-streaming run (used by /chat endpoint) ────────────────────────────────

async def run(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Returns (final_response, steps)."""
    steps: list[dict] = []
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        *history,
        {"role": "user", "content": user_message},
    ]

    provider_used = _clients[0]["name"] if _clients else "none"

    for _ in range(MAX_TOOL_ITERATIONS):
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
            final = _TOOL_CALL_RE.sub("", content)
            final = _THINK_RE.sub("", final).strip() or content
            log.info("run: done via %s (%d chars)", provider_used, len(final))
            if provider_used != _clients[0]["name"]:
                final = f"[{provider_used} fallback]\n{final}"
            return final, steps

        result_blocks = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {})
            log.info("run: tool_call %s %s", name, args)
            steps.append({"type": "tool_call",   "name": name, "args": args})
            result = _run_tool(name, args)
            log.debug("run: tool_result %s → %.120s", name, result)
            steps.append({"type": "tool_result", "name": name, "result": result})

            if getattr(msg, "tool_calls", None):
                tc_id = next((tc.id for tc in msg.tool_calls if tc.function.name == name), None)
                if tc_id:
                    result_blocks.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            else:
                result_blocks.append(f'<tool_result name="{name}">{result}</tool_result>')

        if getattr(msg, "tool_calls", None):
            messages.extend(result_blocks)
        else:
            messages.append({"role": "user", "content": "\n".join(result_blocks)})

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
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        *history,
        {"role": "user", "content": user_message},
    ]

    provider_used = _clients[0]["name"] if _clients else "none"

    for iteration in range(MAX_TOOL_ITERATIONS):
        stream, provider = await _call_stream(messages)
        provider_used    = provider
        log.info("stream[%d]: provider=%s", iteration, provider)

        # Collect the full stream, deciding mode on first meaningful delta
        content_parts   = []
        tool_calls_acc  = {}   # index → {id, name, arguments}
        native_tc_list  = []   # for building message history
        tool_mode       = None  # None | "text" | "tools"
        stripper        = _ThinkStripper()

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Tool call deltas
            if delta.tool_calls:
                tool_mode = "tools"
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

            # Content deltas — only forward if we haven't entered tool mode
            if delta.content and tool_mode != "tools":
                tool_mode = "text"
                content_parts.append(delta.content)
                forwarded = stripper.feed(delta.content)
                if forwarded:
                    yield {"type": "text_chunk", "text": forwarded}

        # Flush any remaining stripper buffer
        tail = stripper.finalize()
        if tail and tool_mode == "text":
            yield {"type": "text_chunk", "text": tail}
            content_parts.append(tail)

        full_content = "".join(content_parts)

        if tool_calls_acc:
            # Build native tool_calls list for message history
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

            # Execute tool calls and yield events
            result_messages = []
            for idx in sorted(tool_calls_acc.keys()):
                tc   = tool_calls_acc[idx]
                name = tc["name"]
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                log.info("stream: tool_call %s %s", name, args)
                yield {"type": "tool_call", "name": name, "args": args}
                result = _run_tool(name, args)
                log.debug("stream: tool_result %s → %.120s", name, result)
                yield {"type": "tool_result", "name": name, "result": result}

                result_messages.append({
                    "role":        "tool",
                    "tool_call_id": tc["id"],
                    "content":     result,
                })

            messages.extend(result_messages)
            # Loop for next LLM call

        else:
            # No native tool calls — check XML fallback in accumulated content
            xml_calls = _parse_tool_calls(full_content)
            if xml_calls:
                messages.append({"role": "assistant", "content": full_content})
                result_parts = []
                for call in xml_calls:
                    name = call.get("name", "")
                    args = call.get("args", {})
                    log.info("stream: xml tool_call %s %s", name, args)
                    yield {"type": "tool_call", "name": name, "args": args}
                    result = _run_tool(name, args)
                    log.debug("stream: xml tool_result %s → %.120s", name, result)
                    yield {"type": "tool_result", "name": name, "result": result}
                    result_parts.append(f'<tool_result name="{name}">{result}</tool_result>')
                messages.append({"role": "user", "content": "\n".join(result_parts)})
                # Loop for next LLM call
            else:
                # Truly final text response
                yield {"type": "done", "provider": provider_used}
                return

    yield {"type": "error", "detail": "Reached max tool iterations."}
