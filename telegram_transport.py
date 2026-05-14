"""Telegram long-polling transport (issue #60).

Session IDs:
  ``tg:<chat_id>`` or ``tg:<chat_id>:<message_thread_id>`` for forum topics.
  ``/new`` appends ``:s-`` and a **short typable** slug of lowercase **letters only** (default 5
  characters, a–z) so you can /switch to it without retyping the full id.

``sendMessage`` uses ``parse_mode: HTML``; all dynamic text is entity-escaped.

Requires ``TELEGRAM_BOT_TOKEN`` in the environment. Started from ``main.py`` lifespan
when the token is set.
"""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
import string
from contextlib import suppress
from typing import Any

import httpx

import config
import main

log = logging.getLogger(__name__)

# (chat_id, thread_key) -> active session_id. thread_key is 0 if not a forum thread.
_active_session: dict[tuple[int, int], str] = {}

_TELEGRAM_MSG_LIMIT = 4096
_SESSION_SLUG_LEN = 5
# Lowercase a–z only, short to type, no digits.
_ALPH = string.ascii_lowercase
_PARSE_MODE = "HTML"


# Defensive Telegram-side cleanup only for malformed leftovers (e.g. "<thought"
# with no closing '>'). Normal closed thought blocks are stripped in agent.py.
_THINK_MALFORMED_OPEN_RE = re.compile(
    r"<(?:thought|think|thinking|redacted_reasoning|redacted_thinking)\b[^>]*$",
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(
    r"<(thought|think|thinking|redacted_reasoning|redacted_thinking)[\s>].*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_TAG_RE = re.compile(
    r"</?(thought|think|thinking|redacted_reasoning|redacted_thinking)\b[^>]*>",
    re.IGNORECASE,
)


def _html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _agent_reply_for_tg(s: str) -> str:
    """Sanitize model text for Telegram send/edit.

    Normal path should already strip thoughts in agent stream. This keeps a
    minimal defense-in-depth layer for Telegram rendering.
    """
    t = s or ""
    t = _THINK_BLOCK_RE.sub("", t)
    t = _THINK_TAG_RE.sub("", t)
    t = _THINK_MALFORMED_OPEN_RE.sub("", t)
    t = t.replace("**", "")
    return _html_escape(t)


def _random_letter_slug(length: int = _SESSION_SLUG_LEN) -> str:
    return "".join(secrets.choice(_ALPH) for _ in range(length))


def _session_branch_slug(session_id: str) -> str | None:
    """The letters-only part after the last ``:s-``, or None for the default branch (no ``:s-``)."""
    if ":s-" not in session_id:
        return None
    return session_id.rsplit(":s-", 1)[-1]


def _user_facing_label(session_id: str) -> str:
    """Short name for Telegram copy only (slug or 'default'). Full id stays in SQLite."""
    sl = _session_branch_slug(session_id)
    return sl if sl is not None else "default"


def _sender_user_id(message: dict[str, Any]) -> int | None:
    from_user = message.get("from")
    if not from_user or from_user.get("id") is None:
        return None
    return int(from_user["id"])


def _is_sender_allowed(user_id: int | None) -> bool:
    allowed = config.TELEGRAM_ALLOWED_USER_IDS
    if allowed is None:
        return True
    if user_id is None:
        return False
    return user_id in allowed


def _thread_key(message: dict[str, Any]) -> int:
    tid = message.get("message_thread_id")
    return int(tid) if tid is not None else 0


def _default_session_id(chat_id: int, thread_key: int) -> str:
    if thread_key:
        return f"tg:{chat_id}:{thread_key}"
    return f"tg:{chat_id}"


def _prefs_key(chat_id: int, thread_key: int) -> tuple[int, int]:
    return (chat_id, thread_key)


def _new_session_id(chat_id: int, thread_key: int) -> str:
    slug = _random_letter_slug()
    base = _default_session_id(chat_id, thread_key)
    return f"{base}:s-{slug}"


def _active_sid(chat_id: int, thread_key: int) -> str:
    k = _prefs_key(chat_id, thread_key)
    if k not in _active_session:
        _active_session[k] = _default_session_id(chat_id, thread_key)
    return _active_session[k]


def _set_active(chat_id: int, thread_key: int, session_id: str) -> None:
    _active_session[_prefs_key(chat_id, thread_key)] = session_id


def _resolve_session_switch(
    want: str,
    valid: set[str],
) -> str | None:
    """Map user input to a session_id: exact id, or unique short :s- slug (letters only)."""
    w = (want or "").strip()
    if not w:
        return None
    if w in valid:
        return w
    if re.fullmatch(r"[a-z]+", w) and 2 <= len(w) <= 12:
        cands = [s for s in valid if (sl := _session_branch_slug(s)) is not None and sl == w]
        if len(cands) == 1:
            return cands[0]
    if m := re.fullmatch(r"s-([a-z]+)", w):
        inner = m.group(1)
        cands2 = [s for s in valid if (sl := _session_branch_slug(s)) is not None and sl == inner]
        if len(cands2) == 1:
            return cands2[0]
    return None


def _telegram_sessions_for_chat(chat_id: int, thread_key: int) -> list[dict]:
    """Sessions that belong to this Telegram chat (and optional thread)."""
    base = _default_session_id(chat_id, thread_key)
    prefix = base + ":"
    out: list[dict] = []
    for s in main.db.get_sessions():
        sid = s["session_id"]
        if sid == base or sid.startswith(prefix):
            out.append(s)
    out.sort(key=lambda x: int(x.get("last_ts") or 0), reverse=True)
    return out


async def _send_chat_action(
    client: httpx.AsyncClient, api: str, chat_id: int, *, thread_key: int, action: str = "typing"
) -> None:
    body: dict[str, Any] = {"chat_id": chat_id, "action": action}
    if thread_key:
        body["message_thread_id"] = thread_key
    with suppress(Exception):
        await client.post(f"{api}/sendChatAction", json=body)


async def _typing_loop(
    client: httpx.AsyncClient, api: str, chat_id: int, *, thread_key: int
) -> None:
    """Re-send typing every 4 s until cancelled (Telegram indicator expires after ~5 s)."""
    try:
        while True:
            await _send_chat_action(client, api, chat_id, thread_key=thread_key)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def _send_text(
    client: httpx.AsyncClient,
    api: str,
    chat_id: int,
    text: str,
    *,
    thread_key: int,
    parse_mode: str | None = _PARSE_MODE,
) -> None:
    """Send rich text. ``text`` must be valid Telegram HTML (we escape where needed in callers)."""
    text = text or ""
    params: dict[str, Any] = {"chat_id": chat_id}
    if thread_key:
        params["message_thread_id"] = thread_key
    if parse_mode:
        params["parse_mode"] = parse_mode
    while text:
        chunk = text[:_TELEGRAM_MSG_LIMIT]
        text = text[_TELEGRAM_MSG_LIMIT:]
        r = await client.post(f"{api}/sendMessage", json={**params, "text": chunk})
        if r.status_code != 200:
            log.warning("telegram sendMessage failed: %s %s", r.status_code, r.text[:500])


async def _send_text_get_message_id(
    client: httpx.AsyncClient,
    api: str,
    chat_id: int,
    text: str,
    *,
    thread_key: int,
    parse_mode: str | None = _PARSE_MODE,
) -> int | None:
    """Send one message and return Telegram message_id (or None on failure)."""
    body: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if thread_key:
        body["message_thread_id"] = thread_key
    if parse_mode:
        body["parse_mode"] = parse_mode
    r = await client.post(f"{api}/sendMessage", json=body)
    if r.status_code != 200:
        log.warning("telegram sendMessage failed: %s %s", r.status_code, r.text[:500])
        return None
    try:
        data = r.json()
        if data.get("ok") and data.get("result", {}).get("message_id") is not None:
            return int(data["result"]["message_id"])
    except Exception:
        pass
    return None


async def _edit_text(
    client: httpx.AsyncClient,
    api: str,
    chat_id: int,
    message_id: int,
    text: str,
    *,
    thread_key: int,
    parse_mode: str | None = _PARSE_MODE,
) -> None:
    """Best-effort edit for streaming preview updates."""
    body: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if thread_key:
        body["message_thread_id"] = thread_key
    if parse_mode:
        body["parse_mode"] = parse_mode
    with suppress(Exception):
        await client.post(f"{api}/editMessageText", json=body)


def _parse_command(text: str) -> tuple[str | None, str]:
    t = (text or "").strip()
    if not t.startswith("/"):
        return None, t
    rest = t[1:]
    if not rest.strip():
        return "", ""
    parts = rest.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]
    return cmd.lower(), arg.strip()


async def _handle_command(
    client: httpx.AsyncClient,
    api: str,
    chat_id: int,
    thread_key: int,
    cmd: str,
    arg: str,
) -> None:
    if not cmd:
        cmd = "help"

    if cmd == "new":
        sid = _new_session_id(chat_id, thread_key)
        _set_active(chat_id, thread_key, sid)
        label = _html_escape(_user_facing_label(sid))
        body = f"New session.\nActive: <b>{label}</b>\nSend a message to start."
        await _send_text(client, api, chat_id, body, thread_key=thread_key)
        return

    if cmd == "sessions":
        rows = _telegram_sessions_for_chat(chat_id, thread_key)
        cur = _active_sid(chat_id, thread_key)
        if not rows:
            await _send_text(client, api, chat_id, "No saved sessions for this chat yet.", thread_key=thread_key)
            return
        cur_label = _html_escape(_user_facing_label(cur))
        lines: list[str] = [f"Active: <b>{cur_label}</b>", ""]
        for i, s in enumerate(rows[:20], start=1):
            mark = " ← active" if s["session_id"] == cur else ""
            full = s["session_id"]
            label = _html_escape(_user_facing_label(full))
            pv = _html_escape((s.get("preview") or "").replace("\n", " "))
            lines.append(f"{i}. <b>{label}</b> — {pv}{mark}")
        if len(rows) > 20:
            lines.append(f"\n… and {len(rows) - 20} more")
        await _send_text(client, api, chat_id, "\n".join(lines), thread_key=thread_key)
        return

    if cmd == "switch":
        if not arg:
            await _send_text(
                client,
                api,
                chat_id,
                "Usage: /switch &lt;name&gt; (see /sessions).",
                thread_key=thread_key,
            )
            return
        want_in = arg.strip()
        valid_list = _telegram_sessions_for_chat(chat_id, thread_key)
        valid = {s["session_id"] for s in valid_list}
        resolved = _resolve_session_switch(want_in, valid)
        if resolved is None:
            h = _html_escape(want_in)
            await _send_text(
                client,
                api,
                chat_id,
                f"Unknown or ambiguous: <b>{h}</b>\nCheck /sessions for the name to use.",
                thread_key=thread_key,
            )
            return
        _set_active(chat_id, thread_key, resolved)
        sw_label = _html_escape(_user_facing_label(resolved))
        await _send_text(
            client, api, chat_id, f"Switched to <b>{sw_label}</b>.", thread_key=thread_key
        )
        return

    if cmd == "nuke":
        sid = _active_sid(chat_id, thread_key)
        hint = arg.strip()
        extra = f"\n\nUser note: {hint}" if hint else ""
        await _send_text(client, api, chat_id, "Nuking session (summarizing and resetting)…", thread_key=thread_key)
        typing = asyncio.create_task(_typing_loop(client, api, chat_id, thread_key=thread_key))
        try:
            reply, _ = await main.complete_chat_turn(
                "Use the nuke_chat tool now to reset this conversation. "
                "The summary must preserve everything important for continuing work."
                + extra,
                sid,
                output_channel="telegram",
            )
        except Exception as e:
            typing.cancel()
            log.exception("telegram /nuke failed")
            await _send_text(
                client, api, chat_id, f"Error: {_html_escape(str(e))}", thread_key=thread_key
            )
            return
        typing.cancel()
        await _send_text(client, api, chat_id, _agent_reply_for_tg(reply), thread_key=thread_key)
        return

    if cmd in ("start", "help"):
        a = _active_sid(chat_id, thread_key)
        a_label = _html_escape(_user_facing_label(a))
        help_body = (
            "Commands:\n"
            "/new — new session\n"
            "/sessions — list sessions\n"
            "/switch &lt;name&gt;\n"
            "/nuke — summarize &amp; reset current session\n"
            "/help — this text\n\n"
            f"Active: <b>{a_label}</b>"
        )
        await _send_text(client, api, chat_id, help_body, thread_key=thread_key)
        return

    hcmd = _html_escape(cmd)
    await _send_text(
        client, api, chat_id, f"Unknown command /{hcmd}. Try /help.", thread_key=thread_key
    )


async def _handle_message(client: httpx.AsyncClient, api: str, message: dict[str, Any]) -> None:
    uid = _sender_user_id(message)
    if not _is_sender_allowed(uid):
        log.debug("telegram: ignored message from user %s (not in TELEGRAM_ALLOWED_USER_IDS)", uid)
        return

    chat = message.get("chat") or {}
    chat_id = int(chat["id"])
    thread_key = _thread_key(message)
    text = (message.get("text") or "").strip()
    if not text:
        await _send_text(
            client, api, chat_id, "<i>Only text messages are supported.</i>", thread_key=thread_key
        )
        return

    cmd, arg = _parse_command(text)
    if cmd is not None:
        await _handle_command(client, api, chat_id, thread_key, cmd, arg)
        return

    sid = _active_sid(chat_id, thread_key)
    # Start with a tiny placeholder and stream progress via edits.
    message_id = await _send_text_get_message_id(
        client, api, chat_id, "…", thread_key=thread_key, parse_mode=None
    )

    history = main._get_history(sid)
    # Immediate persistence to ensure durability.
    pending_row_id = main.db.create_pending_turn(sid, text)
    state = main._StreamTurnState(sid, text, [], pending_row_id)
    producer = asyncio.create_task(
        main._run_stream_turn(state, text, history, output_channel="telegram")
    )
    typing = asyncio.create_task(_typing_loop(client, api, chat_id, thread_key=thread_key))

    partial = ""
    last_sent = ""
    last_edit_ts = 0.0
    try:
        while True:
            ev = await state.queue.get()
            if ev is None:
                break
            if ev.get("type") == "done":
                # Stop typing as soon as model generation is done; no need to
                # keep indicator alive while persistence/finalization drains.
                typing.cancel()
            if ev.get("type") == "text_chunk":
                partial += ev.get("text", "")
                # Telegram edit throttling + size guard for in-progress preview.
                now = asyncio.get_running_loop().time()
                if message_id is not None and (now - last_edit_ts) >= 0.8:
                    preview = _agent_reply_for_tg(partial)
                    if len(preview) > 3500:
                        preview = preview[:3500] + "…"
                    if preview and preview != last_sent:
                        await _edit_text(
                            client, api, chat_id, message_id, preview, thread_key=thread_key
                        )
                        last_sent = preview
                        last_edit_ts = now
            elif ev.get("type") == "error":
                detail = _html_escape(ev.get("detail") or "stream error")
                if message_id is not None:
                    await _edit_text(
                        client,
                        api,
                        chat_id,
                        message_id,
                        f"Error: {detail}",
                        thread_key=thread_key,
                    )
                else:
                    await _send_text(
                        client, api, chat_id, f"Error: {detail}", thread_key=thread_key
                    )
                typing.cancel()
        await producer
    except Exception as e:
        typing.cancel()
        log.exception("telegram chat failed")
        producer.cancel()
        with suppress(Exception):
            await producer
        await _send_text(
            client, api, chat_id, f"Error: {_html_escape(str(e))}", thread_key=thread_key
        )
        return

    typing.cancel()
    final = _agent_reply_for_tg(state.full_response)
    if not final:
        final = "(No visible response from the model.)"

    # Finalize preview message; if too long, keep preview and send full as follow-up chunks.
    if message_id is not None:
        if len(final) <= _TELEGRAM_MSG_LIMIT:
            await _edit_text(client, api, chat_id, message_id, final, thread_key=thread_key)
        else:
            await _edit_text(
                client,
                api,
                chat_id,
                message_id,
                "Reply is long; sending full output below…",
                thread_key=thread_key,
                parse_mode=None,
            )
            await _send_text(client, api, chat_id, final, thread_key=thread_key)
    else:
        await _send_text(client, api, chat_id, final, thread_key=thread_key)


async def run_telegram_polling() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        return
    api = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
    offset = 0
    log.info("telegram: polling started")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                r = await client.get(
                    api + "/getUpdates",
                    params={"offset": offset, "timeout": 50, "allowed_updates": ["message"]},
                    timeout=60.0,
                )
                data = r.json()
                if not data.get("ok"):
                    log.warning("telegram getUpdates: %s", data)
                    await asyncio.sleep(3)
                    continue
                for u in data.get("result", []):
                    offset = max(offset, int(u["update_id"]) + 1)
                    msg = u.get("message")
                    if not msg:
                        continue
                    try:
                        await _handle_message(client, api, msg)
                    except Exception:
                        log.exception("telegram: handle message failed")
            except asyncio.CancelledError:
                log.info("telegram: polling cancelled")
                raise
            except Exception:
                log.exception("telegram: poll loop error")
                await asyncio.sleep(5)
