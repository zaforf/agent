"""SQLite-backed conversation history.

One row per turn. Each row stores:
- `content`: the user message text (used for session preview queries)
- `turn_messages`: the full message sequence added on this turn (user +
  intermediate assistant w/ tool_calls + tool results + final assistant)

`get_history()` expands `turn_messages` back into a flat message list for the
LLM; `get_display_history()` collapses the same data into user/assistant pairs
with tool steps synthesized from tool_calls/tool messages for the UI.
"""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "history.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT    NOT NULL,
                role           TEXT    NOT NULL,
                content        TEXT    NOT NULL,
                turn_messages  TEXT    DEFAULT NULL,
                ts             INTEGER DEFAULT (unixepoch())
            )
        """)


def append_turn(session_id: str, user_message: str, turn_messages: list[dict]) -> int:
    """Store a complete turn as a single row. Returns the inserted row ID.

    turn_messages is the full slice starting from the user message through to
    the final assistant reply, including all intermediate tool-call and
    tool-result messages. get_history() expands this back into the full
    sequence when building LLM context; get_display_history() collapses it
    into UI-shaped user/assistant pairs.

    The row ID is returned so callers can later `update_turn_messages()` the
    same row — this is how post-turn summarization patches the stored form
    after the response has already been delivered to the user (DESIGN §6.5).
    """
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO messages (session_id, role, content, turn_messages) VALUES (?, ?, ?, ?)",
            (session_id, "user", user_message, json.dumps(turn_messages)),
        )
        return cur.lastrowid


def update_turn_messages(row_id: int, turn_messages: list[dict]) -> None:
    """Overwrite the stored `turn_messages` blob for a single row.

    Used by the post-turn summary finalizer in main.py: after a turn completes
    with unfinished summary tasks, the finalizer awaits them, patches the
    in-memory `turn_messages` in place, then calls this to make the change
    durable.
    """
    with _conn() as c:
        c.execute(
            "UPDATE messages SET turn_messages = ? WHERE id = ?",
            (json.dumps(turn_messages), row_id),
        )


def get_history(session_id: str) -> list[dict]:
    """Flat message list for LLM context, concatenated across all turns."""
    with _conn() as c:
        rows = c.execute(
            "SELECT turn_messages FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    result: list[dict] = []
    for r in rows:
        tm_raw = r["turn_messages"]
        if not tm_raw:
            continue
        try:
            result.extend(json.loads(tm_raw))
        except Exception:
            continue
    return result


def get_display_history(session_id: str) -> list[dict]:
    """Return conversation history in a display-friendly format for the frontend.

    Collapses the full message sequence (user, intermediate assistant with tool_calls,
    tool results, final assistant) into paired user/assistant messages where tool
    call/result pairs are attached as `steps` on the assistant message.

    The LLM always receives the full sequence via get_history(); this function is
    only for rendering the conversation in the UI.
    """
    messages = get_history(session_id)
    result: list[dict] = []
    pending_tcs: dict = {}   # tool_call_id → tool_call dict
    pending_steps: list = [] # accumulated steps for the current assistant turn
    current_user: dict | None = None

    for msg in messages:
        role = msg.get("role")

        if role == "user":
            content = msg["content"]
            if isinstance(content, list):
                # Multimodal turn — flatten to displayable text for the UI.
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        parts.append("[image]")
                content = " ".join(parts)
            current_user = {"role": "user", "content": content}
            pending_tcs = {}
            pending_steps = []

        elif role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                pending_tcs[tc["id"]] = tc

        elif role == "tool":
            tc = pending_tcs.get(msg.get("tool_call_id", ""))
            args: dict = {}
            if tc:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except Exception:
                    pass
            pending_steps.append({
                "type": "tool_call",
                "name": msg.get("name") or (tc["function"]["name"] if tc else "tool"),
                "args": args,
            })
            pending_steps.append({
                "type": "tool_result",
                "name": msg.get("name", "tool"),
                "result": msg.get("content", ""),
            })

        elif role == "assistant":
            if current_user is not None:
                result.append(current_user)
                current_user = None
            entry: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if pending_steps:
                entry["steps"] = list(pending_steps)
                pending_steps = []
            result.append(entry)

    if current_user is not None:
        result.append(current_user)

    return result


def clear(session_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def get_sessions() -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT
                session_id,
                COUNT(*) as count,
                MAX(ts)  as last_ts,
                (SELECT content FROM messages m2
                 WHERE m2.session_id = m.session_id AND m2.role = 'user'
                 ORDER BY id DESC LIMIT 1) as preview
            FROM messages m
            GROUP BY session_id
            ORDER BY last_ts DESC
        """).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "count":      r["count"],
            "last_ts":    r["last_ts"],
            "preview":    (r["preview"] or "")[:72],
        }
        for r in rows
    ]
