import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import agent
import db
from tools.memory import get_all as get_all_memories, delete_memory
from tools.self_modify import get_system_prompt, edit_system_prompt

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


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response:   str
    session_id: str
    provider:   str = ""


# ── Chat (non-streaming, kept for compat / testing) ───────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = _get_history(req.session_id)
    try:
        response, provider, turn_messages, pending = await agent.run(
            req.message, history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.extend(turn_messages)
    row_id = db.append_turn(req.session_id, req.message, turn_messages)
    _spawn_finalizer(pending, turn_messages, row_id)

    return ChatResponse(response=response, session_id=req.session_id, provider=provider)


# ── Chat (streaming SSE) ───────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    history = _get_history(req.session_id)

    async def generate():
        full_response  = ""
        turn_messages: list[dict] = []
        pending: list[tuple[dict, asyncio.Task]] = []

        try:
            async for event in agent.run_stream(req.message, history):
                if event["type"] == "text_chunk":
                    full_response += event["text"]
                elif event["type"] == "done":
                    turn_messages = event.get("turn_messages", [])
                    # Pop tasks before serializing — they are not JSON-safe and
                    # are handed to the background finalizer below.
                    pending = event.pop("pending_summaries", [])

                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            return

        if full_response and turn_messages:
            history.extend(turn_messages)
            row_id = db.append_turn(req.session_id, req.message, turn_messages)
            _spawn_finalizer(pending, turn_messages, row_id)

    return StreamingResponse(generate(), media_type="text/event-stream")


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


# ── System prompt ─────────────────────────────────────────────────────────────

class SystemPromptBody(BaseModel):
    content: str


@app.get("/system-prompt")
def get_sp():
    return {"content": get_system_prompt()}


@app.put("/system-prompt")
def set_sp(body: SystemPromptBody):
    edit_system_prompt(body.content, "Updated via UI")
    return {"ok": True}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
