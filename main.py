import asyncio
import io
import inspect
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from typing import Literal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import agent
import config
import db
from tools.memory import get_all as get_all_memories, delete_memory

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress third-party noise; keep agent + tools logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    tg_task: asyncio.Task | None = None
    if config.TELEGRAM_BOT_TOKEN:
        from telegram_transport import run_telegram_polling

        tg_task = asyncio.create_task(run_telegram_polling(), name="telegram-poll")
    yield
    if tg_task is not None:
        tg_task.cancel()
        with suppress(asyncio.CancelledError):
            await tg_task


app = FastAPI(title="Agent", lifespan=lifespan)

# In-memory cache: session_id → history list
_cache: dict[str, list[dict]] = {}

# Keep refs to background summary-finalizer tasks alive until they complete;
# without this, Python may garbage-collect an in-flight fire-and-forget task.
_background_tasks: set[asyncio.Task] = set()




class _StreamTurnState:
    """Server-owned lifecycle for one streaming turn per session."""

    def __init__(self, session_id: str, req_message: str, display_files: list[dict]):
        self.session_id = session_id
        self.req_message = req_message
        self.display_files = display_files
        self.queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self.turn_messages: list[dict] = []
        self.pending: list[tuple[dict, asyncio.Task]] = []
        self.full_response = ""
        self.nuke_summary: str | None = None
        self.completed = False
        self.was_cancelled = False
        self.task: asyncio.Task | None = None


_active_stream_turns: dict[str, _StreamTurnState] = {}


async def _persist_stream_turn(state: _StreamTurnState) -> None:
    """Persist or reset history after producer completion (disconnect-safe)."""
    history = _get_history(state.session_id)

    if state.nuke_summary is not None:
        _apply_nuke(state.session_id, history, state.nuke_summary)
        return

    if (not state.was_cancelled) and state.full_response and state.turn_messages:
        history.extend(state.turn_messages)
        row_id = db.append_turn(state.session_id, state.req_message, state.turn_messages)
        _spawn_finalizer(state.pending, state.turn_messages, row_id)


async def _run_stream_turn(
    state: _StreamTurnState,
    user_content: "str | list",
    history: list[dict],
    *,
    output_channel: str = "default",
) -> None:
    """Background producer: runs agent stream, queues SSE events, persists on completion."""
    try:
        run_stream_sig = inspect.signature(agent.run_stream)
        kwargs = {"output_channel": output_channel} if "output_channel" in run_stream_sig.parameters else {}
        async for event in agent.run_stream(user_content, history, **kwargs):
            if event.get("type") == "text_chunk":
                state.full_response += event.get("text", "")
            elif event.get("type") == "done":
                state.turn_messages = event.get("turn_messages", [])
                state.pending = event.pop("pending_summaries", [])
                state.nuke_summary = _extract_nuke_summary(state.turn_messages)
                if state.nuke_summary is None:
                    _patch_display_files(state.turn_messages, state.display_files)
            await state.queue.put(event)
    except asyncio.CancelledError:
        state.was_cancelled = True
        await state.queue.put({"type": "cancelled"})
        raise
    except Exception as e:
        await state.queue.put({"type": "error", "detail": str(e)})
    finally:
        with suppress(Exception):
            await _persist_stream_turn(state)
        state.completed = True
        await state.queue.put(None)

def _get_history(session_id: str) -> list[dict]:
    if session_id not in _cache:
        _cache[session_id] = db.get_history(session_id)
    return _cache[session_id]


async def _finalize_summaries(
    pending: list[tuple[dict, asyncio.Task]],
    turn_messages: list[dict],
    row_id: int,
) -> None:
    """Drain pending summary tasks after a turn has already been returned to
    the user, then overwrite the stored row with the summarized content.

    The tool message dicts in `turn_messages` are the same objects referenced
    by `pending[*][0]` and by the session's in-memory history cache, so the
    patch here propagates to every view that still has them.
    """
    try:
        results = await asyncio.gather(
            *(t for _, t in pending), return_exceptions=True
        )
        for (msg_dict, _), summary in zip(pending, results):
            if isinstance(summary, str):
                msg_dict["content"] = summary
        db.update_turn_messages(row_id, turn_messages)
    except Exception:
        log.exception("post-turn summary finalization failed")


def _spawn_finalizer(pending, turn_messages, row_id) -> None:
    """Fire-and-forget wrapper — holds a strong ref so the task isn't GC'd."""
    if not pending:
        return
    task = asyncio.create_task(_finalize_summaries(pending, turn_messages, row_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ── File upload ───────────────────────────────────────────────────────────────

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF (only MIME accepted server-side).

    Text files and images are handled entirely client-side; this endpoint
    exists solely for PDF extraction via pypdf (DESIGN §13).
    """
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 5 MB limit")

    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail=f"Only PDFs are processed server-side; got {content_type!r}",
        )

    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # pypdf can produce lone surrogates from malformed/encoded PDFs; strip them
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF text extraction failed: {e}")

    return {"filename": file.filename, "type": "text", "content": text}


# ── Request / response models ─────────────────────────────────────────────────

class Attachment(BaseModel):
    type:     Literal["text", "image"]
    filename: str
    content:  str   # text string, or "data:<mime>;base64,..." for images


class ChatRequest(BaseModel):
    message:     str
    session_id:  str             = "default"
    attachments: list[Attachment] = []


class StreamCancelRequest(BaseModel):
    session_id: str


def _patch_display_files(turn_messages: list[dict], display_files: list[dict]) -> None:
    """Attach display-only file metadata to the user message in turn_messages.

    Stored as `_display_files` on the user message dict so db.get_display_history()
    can render chips without re-exposing raw file content. The field is stripped by
    agent._sanitize_message (whitelist-based) before it reaches the LLM.
    """
    if not display_files:
        return
    for msg in turn_messages:
        if msg.get("role") == "user":
            msg["_display_files"] = display_files
            return




def _extract_nuke_summary(turn_messages: list[dict]) -> str | None:
    if len(turn_messages) != 1:
        return None
    msg = turn_messages[0]
    if msg.get("role") != "assistant" or not msg.get("_nuke"):
        return None
    return (msg.get("content") or "").strip() or None




def _format_nuke_summary(summary: str) -> str:
    summary = summary.strip()
    return f"Chat reset via nuke. Summary:\n{summary}" if summary else "Chat reset via nuke."

def _apply_nuke(session_id: str, history: list[dict], summary: str) -> None:
    """Replace entire session history with one assistant summary message."""
    formatted = _format_nuke_summary(summary)
    clean_turn = [{"role": "assistant", "content": formatted}]
    db.clear(session_id)
    history.clear()
    history.extend(clean_turn)
    db.append_turn(session_id, formatted, clean_turn)

def _build_user_content(message: str, attachments: list[Attachment]) -> "str | list":
    """Return a plain string when there are no attachments (backward-compat).

    With attachments, return a multimodal content list (OpenAI vision format):
    file context parts first, then the user's text message.
    """
    if not attachments:
        return message
    parts: list[dict] = []
    for att in attachments:
        if att.type == "text":
            parts.append({"type": "text", "text": f"[File: {att.filename}]\n{att.content}"})
        elif att.type == "image":
            parts.append({"type": "image_url", "image_url": {"url": att.content}})
    if message:
        parts.append({"type": "text", "text": message})
    return parts


class ChatResponse(BaseModel):
    response:   str
    session_id: str
    provider:   str = ""


async def complete_chat_turn(
    message: str,
    session_id: str,
    *,
    attachments: list[Attachment] | None = None,
    output_channel: str = "default",
) -> tuple[str, str]:
    """Run one non-streaming agent turn: same persistence rules as ``POST /chat``.

    Returns ``(assistant_visible_text, provider_name)``. Mutates ``_cache`` / SQLite.

    Use ``output_channel="telegram"`` for the Telegram bot (plain-text-friendly system prompt);
    the web UI uses the default.
    """
    attachments = attachments or []
    history = _get_history(session_id)
    user_content = _build_user_content(message, attachments)
    display_files = [{"type": a.type, "filename": a.filename} for a in attachments]
    response, provider, turn_messages, pending = await agent.run(
        user_content, history, output_channel=output_channel
    )

    nuke_summary = _extract_nuke_summary(turn_messages)
    if nuke_summary is not None:
        _apply_nuke(session_id, history, nuke_summary)
        return _format_nuke_summary(nuke_summary), provider

    _patch_display_files(turn_messages, display_files)
    history.extend(turn_messages)
    row_id = db.append_turn(session_id, message, turn_messages)
    _spawn_finalizer(pending, turn_messages, row_id)

    return response, provider


# ── Chat (non-streaming, kept for compat / testing) ───────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        response, provider = await complete_chat_turn(
            req.message, req.session_id, attachments=req.attachments
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(response=response, session_id=req.session_id, provider=provider)


# ── Chat (streaming SSE) ───────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    history = _get_history(req.session_id)
    user_content = _build_user_content(req.message, req.attachments)
    display_files = [{"type": a.type, "filename": a.filename} for a in req.attachments]

    # One active producer per session. A new request for same session while one
    # is running attaches to existing stream events instead of starting duplicate generation.
    state = _active_stream_turns.get(req.session_id)
    if state is None or state.completed:
        state = _StreamTurnState(req.session_id, req.message, display_files)
        state.task = asyncio.create_task(_run_stream_turn(state, user_content, history))
        _active_stream_turns[req.session_id] = state

    async def generate():
        try:
            while True:
                item = await state.queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected; generation continues server-side for durability.
            return
        finally:
            # Cleanup finished states.
            cur = _active_stream_turns.get(req.session_id)
            if cur is state and state.completed:
                _active_stream_turns.pop(req.session_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/stream/cancel")
async def chat_stream_cancel(req: StreamCancelRequest):
    state = _active_stream_turns.get(req.session_id)
    if state is None or state.task is None or state.task.done():
        return {"cancelled": False}
    state.task.cancel()
    return {"cancelled": True}


# ── History / sessions ────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/history")
def session_history(session_id: str):
    return {"messages": db.get_display_history(session_id)}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    _cache.pop(session_id, None)
    db.clear(session_id)
    return {"cleared": session_id}


@app.get("/sessions")
def list_sessions():
    return {"sessions": db.get_sessions()}


# ── Memories ──────────────────────────────────────────────────────────────────

@app.get("/memories")
def list_memories_api():
    return {"memories": get_all_memories()}


@app.delete("/memories/{memory_id}")
def delete_memory_api(memory_id: str):
    try:
        delete_memory(memory_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": memory_id}


# ── Health ────────────────────────────────────────────────────────────────────

class ContextualQueryRequest(BaseModel):
    selection: str
    context:   str
    query:     str = ""


@app.post("/contextual_query")
async def contextual_query_endpoint(req: ContextualQueryRequest):
    from contextual_query import contextual_query
    try:
        result = await contextual_query(req.selection, req.context, req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"result": result}


def _history_to_gemini_contents(history: list[dict]) -> list[dict]:
    """Convert OpenAI-format history to Gemini contents for countTokens."""
    contents = []
    for msg in history:
        role = msg.get("role")
        if role == "system":
            continue
        gemini_role = "model" if role == "assistant" else "user"
        parts: list[dict] = []
        if role == "tool":
            parts = [{"functionResponse": {
                "name": msg.get("name", ""),
                "response": {"output": msg.get("content", "")},
            }}]
        else:
            content = msg.get("content") or ""
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        if t := part.get("text", ""):
                            parts.append({"text": t})
            elif content:
                parts.append({"text": str(content)})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
        if parts:
            contents.append({"role": gemini_role, "parts": parts})
    return contents


@app.get("/tokens/{session_id}")
async def get_token_count(session_id: str):
    import httpx
    system_prompt = agent._build_system_prompt()
    history = _get_history(session_id)
    model = config.PROVIDERS[0]["model"]
    api_key = config.GEMINI_API_KEY
    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"
            body = {
                "generateContentRequest": {
                    "model": f"models/{model}",
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": _history_to_gemini_contents(history),
                }
            }
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, params={"key": api_key}, json=body)
                resp.raise_for_status()
                return {"tokens": resp.json()["totalTokens"]}
        except Exception as e:
            log.warning("countTokens failed, using heuristic: %s", e)
    # Fallback: char heuristic
    total_chars = len(system_prompt)
    for msg in history:
        content = msg.get("content") or ""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
        else:
            total_chars += len(content)
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total_chars += len(fn.get("name", "")) + len(fn.get("arguments", ""))
    return {"tokens": total_chars // 4}

@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
