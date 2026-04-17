"""
Minimal async Gemini client with an OpenAI-compatible interface.

The Gemini OpenAI-compat endpoint (/v1beta/openai/) doesn't accept the new
AI Studio key format (AQ. prefix). The native endpoint does. This adapter
translates between OpenAI message/tool format and the native Gemini REST API,
exposing only the interface that agent.py actually uses.
"""

import asyncio
import json
import uuid
import httpx

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ── Fake response objects that match what agent.py reads ─────────────────────

class _Fn:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class _ToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = _Fn(name, arguments)

class _Message:
    def __init__(self, content: str | None, tool_calls: list):
        self.content = content
        self.tool_calls = tool_calls or []

class _Choice:
    def __init__(self, message: _Message):
        self.message = message

class _Response:
    def __init__(self, choices: list):
        self.choices = choices

# Streaming
class _DeltaFn:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments

class _DeltaToolCall:
    def __init__(self, index: int, id: str, name: str, arguments: str):
        self.index = index
        self.id = id
        self.type = "function"
        self.function = _DeltaFn(name, arguments)

class _Delta:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls

class _StreamChoice:
    def __init__(self, delta: _Delta):
        self.delta = delta

class _Chunk:
    def __init__(self, choices: list):
        self.choices = choices


# ── Format translation ────────────────────────────────────────────────────────

def _to_gemini_contents(messages: list[dict]) -> tuple[list, dict | None]:
    """Convert OpenAI messages to Gemini contents + optional systemInstruction."""
    system_parts = []
    contents = []

    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""

        if role == "system":
            system_parts.append({"text": content})

        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})

        elif role == "assistant":
            parts = []
            if content:
                parts.append({"text": content})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                parts.append({"functionCall": {"name": fn["name"], "args": args}})
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif role == "tool":
            # Group consecutive tool results under one user turn
            result_part = {
                "functionResponse": {
                    "name": m.get("name", "tool"),
                    "response": {"content": m.get("content", "")},
                }
            }
            if contents and contents[-1]["role"] == "user" and \
               any("functionResponse" in p for p in contents[-1]["parts"]):
                contents[-1]["parts"].append(result_part)
            else:
                contents.append({"role": "user", "parts": [result_part]})

    system_instruction = {"parts": system_parts} if system_parts else None
    return contents, system_instruction


def _to_gemini_tools(tools: list) -> list:
    declarations = []
    for t in tools:
        fn = t.get("function", {})
        declarations.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return [{"functionDeclarations": declarations}]


def _parse_candidate(candidate: dict) -> _Choice:
    parts = candidate.get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
    fn_parts = [p["functionCall"] for p in parts if "functionCall" in p]

    content = "".join(text_parts) or None
    tool_calls = [
        _ToolCall(
            id=f"tc_{uuid.uuid4().hex[:8]}",
            name=fc["name"],
            arguments=json.dumps(fc.get("args", {})),
        )
        for fc in fn_parts
    ]
    return _Choice(_Message(content, tool_calls))


# ── Completions ───────────────────────────────────────────────────────────────

class _Completions:
    def __init__(self, api_key: str, model: str):
        self._key = api_key
        self._model = model

    async def create(self, *, model=None, messages, max_tokens=8192,
                     tools=None, tool_choice=None, stream=False):
        mdl = model or self._model
        contents, sys_inst = _to_gemini_contents(messages)
        body: dict = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if sys_inst:
            body["systemInstruction"] = sys_inst
        if tools:
            body["tools"] = _to_gemini_tools(tools)
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        if stream:
            return _GeminiStream(self._key, mdl, body)

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_BASE}/{mdl}:generateContent",
                params={"key": self._key},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        return _Response([_parse_candidate(candidates[0])])


class _GeminiStream:
    """Async iterable that yields _Chunk objects from the streaming endpoint."""

    def __init__(self, api_key: str, model: str, body: dict):
        self._key = api_key
        self._model = model
        self._body = body

    def __aiter__(self):
        return self._generate()

    async def _generate(self):
        url = f"{_BASE}/{self._model}:streamGenerateContent"
        in_thinking = False  # track whether we've opened a <thinking> tag

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", url,
                params={"key": self._key, "alt": "sse"},
                json=self._body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    candidates = data.get("candidates", [])
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])

                    thought_text = "".join(p["text"] for p in parts if "text" in p and p.get("thought"))
                    visible_text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
                    fn_parts    = [p["functionCall"] for p in parts if "functionCall" in p]

                    # Wrap thought parts in <thinking> tags so agent.py's
                    # _ThinkStripper can detect them and emit thinking_chars events.
                    if thought_text:
                        if not in_thinking:
                            yield _Chunk([_StreamChoice(_Delta(content="<thinking>"))])
                            in_thinking = True
                        yield _Chunk([_StreamChoice(_Delta(content=thought_text))])

                    if visible_text or fn_parts:
                        if in_thinking:
                            yield _Chunk([_StreamChoice(_Delta(content="</thinking>"))])
                            in_thinking = False
                        if visible_text:
                            yield _Chunk([_StreamChoice(_Delta(content=visible_text))])
                        for i, fc in enumerate(fn_parts):
                            tc = _DeltaToolCall(
                                index=i,
                                id=f"tc_{uuid.uuid4().hex[:8]}",
                                name=fc["name"],
                                arguments=json.dumps(fc.get("args", {})),
                            )
                            yield _Chunk([_StreamChoice(_Delta(tool_calls=[tc]))])

                # If stream ended mid-thought, close the tag
                if in_thinking:
                    yield _Chunk([_StreamChoice(_Delta(content="</thinking>"))])


# ── Top-level client ──────────────────────────────────────────────────────────

class _Chat:
    def __init__(self, api_key: str, model: str):
        self.completions = _Completions(api_key, model)


class GeminiClient:
    """Drop-in replacement for AsyncOpenAI for native Gemini API calls."""
    def __init__(self, api_key: str, model: str):
        self.chat = _Chat(api_key, model)
