"""SQLite-backed conversation history."""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "history.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT    NOT NULL,
                role           TEXT    NOT NULL,
                content        TEXT    NOT NULL,
                steps          TEXT    DEFAULT NULL,
                turn_messages  TEXT    DEFAULT NULL,
                ts             INTEGER DEFAULT (unixepoch())
            )
        """)
        # Migrate existing DBs that lack columns
        for col, definition in [
            ("steps",         "TEXT DEFAULT NULL"),
            ("turn_messages", "TEXT DEFAULT NULL"),
        ]:
            try:
                c.execute(f"ALTER TABLE messages ADD COLUMN {col} {definition}")
            except Exception:
                pass


def append(session_id: str, role: str, content: str,
           steps: list | None = None) -> None:
    """Legacy single-message append (kept for compat)."""
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, steps) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(steps) if steps else None),
        )


def append_turn(session_id: str, user_message: str, turn_messages: list[dict],
                steps: list | None = None) -> None:
    """Store a complete turn as a single row.

    turn_messages is the full slice starting from the user message through to
    the final assistant reply, including all intermediate tool-call and
    tool-result messages. get_history() will expand this back to the full
    sequence when building LLM context.
    """
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, steps, turn_messages) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                "user",
                user_message,
                json.dumps(steps) if steps else None,
                json.dumps(turn_messages),
            ),
        )


def get_history(session_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, steps, turn_messages FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        tm_raw = r["turn_messages"]
        if tm_raw:
            # New format: one row per turn storing the full message sequence.
            # Expand it directly — user message and all tool call/result/assistant
            # messages are all in here.
            try:
                result.extend(json.loads(tm_raw))
            except Exception:
                result.append({"role": r["role"], "content": r["content"]})
        else:
            # Legacy format: separate user / assistant rows.
            entry: dict = {"role": r["role"], "content": r["content"]}
            if r["steps"]:
                try:
                    entry["steps"] = json.loads(r["steps"])
                except Exception:
                    entry["steps"] = []
            result.append(entry)
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
    result = []
    pending_tcs: dict = {}   # tool_call_id → tool_call dict
    pending_steps: list = [] # accumulated steps for the current assistant turn
    current_user: dict | None = None

    for msg in messages:
        role = msg.get("role")

        if role == "user":
            current_user = {"role": "user", "content": msg["content"]}
            pending_tcs = {}
            pending_steps = []

        elif role == "assistant" and msg.get("tool_calls"):
            # Intermediate: collect tool call metadata for step display; no bubble yet
            for tc in msg["tool_calls"]:
                pending_tcs[tc["id"]] = tc

        elif role == "tool":
            # Pair with its tool_call for args
            tc = pending_tcs.get(msg.get("tool_call_id", ""))
            args: dict = {}
            if tc:
                try:
                    import json as _json
                    args = _json.loads(tc["function"]["arguments"] or "{}")
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
            # Final assistant message — emit the user message (if any) then this one
            if current_user is not None:
                result.append(current_user)
                current_user = None
            entry: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if pending_steps:
                entry["steps"] = list(pending_steps)
                pending_steps = []
            result.append(entry)

    # Trailing user message with no assistant reply (shouldn't normally happen)
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
