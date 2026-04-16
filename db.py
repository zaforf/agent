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
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                steps      TEXT    DEFAULT NULL,
                ts         INTEGER DEFAULT (unixepoch())
            )
        """)
        # Migrate existing DBs that lack the steps column
        try:
            c.execute("ALTER TABLE messages ADD COLUMN steps TEXT DEFAULT NULL")
        except Exception:
            pass


def append(session_id: str, role: str, content: str,
           steps: list | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, steps) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(steps) if steps else None),
        )


def get_history(session_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, steps FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        entry: dict = {"role": r["role"], "content": r["content"]}
        if r["steps"]:
            try:
                entry["steps"] = json.loads(r["steps"])
            except Exception:
                entry["steps"] = []
        result.append(entry)
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
