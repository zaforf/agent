import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import agent
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
    yield


app = FastAPI(title="Agent", lifespan=lifespan)

# In-memory cache: session_id → history list
_cache: dict[str, list[dict]] = {}

# Keep refs to background summary-finalizer tasks alive until they complete;
# without this, Python may garbage-collect an in-flight fire-and-forget task.
_background_tasks: set[asyncio.Task] = set()


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


# ── Chat (non-streaming, kept for compat / testing) ───────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = _get_history(req.session_id)
    user_content = _build_user_content(req.message, req.attachments)
    display_files = [{"type": a.type, "filename": a.filename} for a in req.attachments]
    try:
        response, provider, turn_messages, pending = await agent.run(
            user_content, history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _patch_display_files(turn_messages, display_files)
    history.extend(turn_messages)
    row_id = db.append_turn(req.session_id, req.message, turn_messages)
    _spawn_finalizer(pending, turn_messages, row_id)

    return ChatResponse(response=response, session_id=req.session_id, provider=provider)


# ── Chat (streaming SSE) ───────────────────────────────────────────────────────

_active_stream_tasks: dict[str, asyncio.Task] = {}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    history = _get_history(req.session_id)
    user_content = _build_user_content(req.message, req.attachments)
    display_files = [{"type": a.type, "filename": a.filename} for a in req.attachments]

    async def generate():
        full_response  = ""
        turn_messages: list[dict] = []
        pending: list[tuple[dict, asyncio.Task]] = []
        was_cancelled = False
        this_task = asyncio.current_task()
        if this_task is not None:
            _active_stream_tasks[req.session_id] = this_task

        try:
            async for event in agent.run_stream(user_content, history):
                if event["type"] == "text_chunk":
                    full_response += event["text"]
                elif event["type"] == "done":
                    turn_messages = event.get("turn_messages", [])
                    # Pop tasks before serializing — they are not JSON-safe and
                    # are handed to the background finalizer below.
                    pending = event.pop("pending_summaries", [])
                    _patch_display_files(turn_messages, display_files)

                yield f"data: {json.dumps(event)}\n\n"

        except asyncio.CancelledError:
            was_cancelled = True
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            return
        finally:
            if this_task is not None and _active_stream_tasks.get(req.session_id) is this_task:
                _active_stream_tasks.pop(req.session_id, None)

        if (not was_cancelled) and full_response and turn_messages:
            history.extend(turn_messages)
            row_id = db.append_turn(req.session_id, req.message, turn_messages)
            _spawn_finalizer(pending, turn_messages, row_id)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/stream/cancel")
async def chat_stream_cancel(req: StreamCancelRequest):
    task = _active_stream_tasks.get(req.session_id)
    if task is None or task.done():
        return {"cancelled": False}
    task.cancel()
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

@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
