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


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    yield


app = FastAPI(title="Agent", lifespan=lifespan)

# In-memory cache: session_id → history list
_cache: dict[str, list[dict]] = {}


def _get_history(session_id: str) -> list[dict]:
    if session_id not in _cache:
        _cache[session_id] = db.get_history(session_id)
    return _cache[session_id]


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response:   str
    session_id: str
    steps:      list[dict] = []


# ── Chat (non-streaming, kept for compat / testing) ───────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = _get_history(req.session_id)
    try:
        response, steps = await agent.run(req.message, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user",      "content": req.message})
    history.append({"role": "assistant", "content": response})
    db.append(req.session_id, "user",      req.message)
    db.append(req.session_id, "assistant", response, steps=steps or None)

    return ChatResponse(response=response, session_id=req.session_id, steps=steps)


# ── Chat (streaming SSE) ───────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    history = _get_history(req.session_id)

    async def generate():
        full_response = ""
        steps: list[dict] = []

        try:
            async for event in agent.run_stream(req.message, history):
                if event["type"] == "text_chunk":
                    full_response += event["text"]
                elif event["type"] in ("tool_call", "tool_result"):
                    steps.append(event)

                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            return

        # Persist once stream is complete
        if full_response:
            history.append({"role": "user",      "content": req.message})
            history.append({"role": "assistant", "content": full_response})
            db.append(req.session_id, "user",      req.message)
            db.append(req.session_id, "assistant", full_response,
                      steps=steps if steps else None)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── History / sessions ────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/history")
def session_history(session_id: str):
    return {"messages": db.get_history(session_id)}


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
