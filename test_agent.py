"""
Agent test harness.
Run: ./venv/bin/python test_agent.py

Covers:
  1. Config — providers, keys, paths
  2. Agent helpers — visible-after-think, history sanitization, auto-summary path
  3. DB — append / get_history / clear / get_sessions / preview = latest user
  4. Memory — remember / recall (needs Qdrant)
  5. System prompt — read, edit roundtrip
  6. Provider client construction
  7. End-to-end chat + streaming
"""

import sys
import os
import uuid

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, str, str]] = []


def ok(name: str) -> None:
    results.append((name, PASS, ""))
    print(f"  {PASS}  {name}")


def fail(name: str, detail: str) -> None:
    results.append((name, FAIL, detail))
    print(f"  {FAIL}  {name}\n       {detail}")


def skip(name: str, reason: str) -> None:
    results.append((name, SKIP, reason))
    print(f"  {SKIP}  {name} — {reason}")


def run_test(name: str, fn):
    try:
        fn()
        ok(name)
    except AssertionError as e:
        fail(name, str(e))
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}")


# ── 1. Config ─────────────────────────────────────────────────────────────────

print("\n== 1. Config ==")

def test_providers_defined():
    from config import PROVIDERS
    assert len(PROVIDERS) >= 2, f"Expected ≥2 providers, got {len(PROVIDERS)}"
    for p in PROVIDERS:
        for field in ("name", "api_key", "base_url", "model"):
            assert field in p, f"Provider {p.get('name','?')} missing field '{field}'"

def test_gemma4_31b_is_first():
    from config import PROVIDERS
    assert PROVIDERS[0]["model"] == "gemma-4-31b-it", (
        f"Expected gemma-4-31b-it first, got {PROVIDERS[0]['model']}"
    )

def test_gemma4_26b_is_second():
    from config import PROVIDERS
    assert PROVIDERS[1]["model"] == "gemma-4-26b-a4b-it", (
        f"Expected gemma-4-26b-a4b-it second, got {PROVIDERS[1]['model']}"
    )

def test_gemini_key_present():
    from config import GEMINI_API_KEY
    assert GEMINI_API_KEY, "GEMINI_API_KEY is not set in .env"

def test_paths_exist():
    from config import DATA_DIR, SYSTEM_PROMPT_PATH
    assert DATA_DIR.exists(), f"DATA_DIR missing: {DATA_DIR}"
    assert SYSTEM_PROMPT_PATH.exists(), f"system_prompt.md missing: {SYSTEM_PROMPT_PATH}"

for fn in [test_providers_defined, test_gemma4_31b_is_first, test_gemma4_26b_is_second,
           test_gemini_key_present, test_paths_exist]:
    run_test(fn.__name__, fn)

# ── 2. Agent helpers ──────────────────────────────────────────────────────────

print("\n== 2. Agent helpers ==")

def test_visible_after_think_strips_block():
    from agent import _visible_after_think
    s = "<thought>\nscratch\n</thought>\nHello"
    assert _visible_after_think(s) == "Hello"

def test_visible_after_think_multiline():
    from agent import _visible_after_think
    s = "<thinking>x</thinking>\n\nAnswer here."
    assert _visible_after_think(s) == "Answer here."

def test_visible_after_think_empty_when_only_block():
    from agent import _visible_after_think
    s = "<thought>only inside</thought>"
    assert _visible_after_think(s) == ""

def test_sanitize_history_drops_steps():
    from agent import _sanitize_history
    h = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo", "steps": [{"type": "tool_call", "name": "recall"}]},
    ]
    out = _sanitize_history(h)
    assert len(out) == 2
    assert "steps" not in out[1]
    assert out[1]["content"] == "yo"

def test_sanitize_history_keeps_tool_messages():
    from agent import _sanitize_history
    h = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "recall", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "memories"},
    ]
    out = _sanitize_history(h)
    assert len(out) == 2
    assert out[1]["role"] == "tool"
    assert out[1]["content"] == "memories"

for fn in [test_visible_after_think_strips_block, test_visible_after_think_multiline,
           test_visible_after_think_empty_when_only_block, test_sanitize_history_drops_steps,
           test_sanitize_history_keeps_tool_messages]:
    run_test(fn.__name__, fn)

# ── 3. DB ─────────────────────────────────────────────────────────────────────

print("\n== 3. DB ==")

def test_db_round_trip():
    import db
    sid = f"test-{uuid.uuid4().hex[:8]}"
    db.init()
    db.append(sid, "user", "hello from test")
    db.append(sid, "assistant", "hi back")
    history = db.get_history(sid)
    assert len(history) == 2, f"Expected 2 messages, got {len(history)}"
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello from test"
    assert history[1]["role"] == "assistant"
    db.clear(sid)
    assert db.get_history(sid) == [], "Expected empty after clear"

def test_db_sessions_list():
    import db
    sid = f"test-{uuid.uuid4().hex[:8]}"
    db.init()
    db.append(sid, "user", "session list test")
    sessions = db.get_sessions()
    ids = [s["session_id"] for s in sessions]
    assert sid in ids, f"{sid} not in sessions"
    db.clear(sid)

def test_db_sessions_preview_is_latest_user():
    """Preview text should be the most recent user message, not the first."""
    import db
    sid = f"test-{uuid.uuid4().hex[:8]}"
    db.init()
    db.append(sid, "user", "older user line")
    db.append(sid, "assistant", "reply")
    db.append(sid, "user", "newest user preview text")
    by_id = {s["session_id"]: s for s in db.get_sessions()}
    assert sid in by_id, sid
    prev = by_id[sid]["preview"] or ""
    assert "newest" in prev, f"Expected latest user in preview, got {prev!r}"
    assert "older" not in prev, f"Preview should not be first user: {prev!r}"
    db.clear(sid)

for fn in [test_db_round_trip, test_db_sessions_list, test_db_sessions_preview_is_latest_user]:
    run_test(fn.__name__, fn)

# ── 4. Memory (Qdrant) ────────────────────────────────────────────────────────

print("\n== 4. Memory (requires Qdrant) ==")

def _qdrant_running() -> bool:
    try:
        import socket
        from config import QDRANT_HOST, QDRANT_PORT
        s = socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=1)
        s.close()
        return True
    except Exception:
        return False

if not _qdrant_running():
    for name in ["remember_and_recall", "list_and_delete_memory"]:
        skip(name, "Qdrant not reachable on localhost:6333")
else:
    def test_remember_and_recall():
        from tools.memory import remember, recall
        store_result = remember("Zafir studies machine learning and deep learning", category="fact")
        assert "Stored" in store_result or store_result, f"remember() returned unexpected: {store_result}"
        recall_result = recall("machine learning")
        assert recall_result and recall_result != "No relevant memories found.", (
            f"recall() returned nothing: {recall_result}"
        )

    def test_list_and_delete_memory():
        import re
        from tools.memory import remember, list_memories, delete_memory
        remember("Zafir prefers Python for scripting tasks", category="preference")
        listing = list_memories()
        assert listing and listing != "No memories stored.", f"Listing empty: {listing!r}"
        ids = re.findall(r'\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]', listing)
        assert ids, f"No IDs found in listing output:\n{listing}"
        result = delete_memory(ids[-1])
        assert "Deleted" in result or "deleted" in result, f"Delete didn't confirm: {result}"

    for fn in [test_remember_and_recall, test_list_and_delete_memory]:
        run_test(fn.__name__, fn)

# ── 5. System prompt ──────────────────────────────────────────────────────────

print("\n== 5. System prompt ==")

def test_system_prompt_read():
    from tools.self_modify import get_system_prompt
    content = get_system_prompt()
    assert isinstance(content, str) and len(content) > 10, "System prompt too short or wrong type"

def test_system_prompt_roundtrip():
    from tools.self_modify import get_system_prompt, edit_system_prompt
    original = get_system_prompt()
    marker = f"# TEST MARKER {uuid.uuid4().hex[:6]}"
    edit_system_prompt(original + f"\n{marker}\n", "test roundtrip")
    updated = get_system_prompt()
    assert marker in updated, "Marker not found after edit"
    edit_system_prompt(original, "restore after test")
    assert get_system_prompt() == original, "Restore failed"

for fn in [test_system_prompt_read, test_system_prompt_roundtrip]:
    run_test(fn.__name__, fn)

# ── 6. Provider client construction ──────────────────────────────────────────

print("\n== 6. Provider clients ==")

def test_clients_built_for_keyed_providers():
    from agent import _clients
    from config import PROVIDERS
    keyed = [p for p in PROVIDERS if p["api_key"]]
    assert len(_clients) == len(keyed), (
        f"Expected {len(keyed)} clients (providers with keys), got {len(_clients)}"
    )

def test_client_order_matches_provider_chain():
    from agent import _clients
    from config import PROVIDERS
    keyed_names = [p["name"] for p in PROVIDERS if p["api_key"]]
    client_names = [c["name"] for c in _clients]
    assert client_names == keyed_names, (
        f"Client order mismatch.\n  Expected: {keyed_names}\n  Got:      {client_names}"
    )

def test_primary_is_gemma4_31b():
    from agent import _clients
    assert _clients, "No clients built"
    assert _clients[0]["model"] == "gemma-4-31b-it", (
        f"Expected primary model gemma-4-31b-it, got {_clients[0]['model']}"
    )

def test_no_summarize_tool_registered():
    from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
    assert "summarize_text" not in TOOL_FUNCTIONS
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert "summarize_text" not in names

def test_auto_summary_returns_short_for_long_text():
    import asyncio
    import agent
    long_text = "A" * (agent._TOOL_SUMMARY_THRESHOLD + 200)
    out = asyncio.run(agent._summarize_tool_result_if_needed("fetch_url", long_text, "test"))
    assert isinstance(out, str) and len(out) > 0
    assert len(out) < len(long_text), "Expected summarized output to be shorter"

for fn in [test_clients_built_for_keyed_providers, test_client_order_matches_provider_chain,
           test_primary_is_gemma4_31b, test_no_summarize_tool_registered,
           test_auto_summary_returns_short_for_long_text]:
    run_test(fn.__name__, fn)

# ── 7. End-to-end chat ────────────────────────────────────────────────────────

print("\n== 7. End-to-end chat ==")

def test_e2e_simple():
    import asyncio, agent
    response, steps = asyncio.run(agent.run("Reply with exactly: PING_OK", []))
    assert isinstance(response, str) and len(response) > 0, "Empty response"
    assert "PING_OK" in response, f"Expected 'PING_OK' in response, got: {response!r}"

def test_e2e_with_tool_call():
    import asyncio, agent
    response, steps = asyncio.run(agent.run(
        "Use recall to look up anything about Zafir, then give a one-sentence answer about what you found or didn't find.",
        []
    ))
    assert isinstance(response, str) and len(response) > 0, "Empty response"
    tool_calls = [s for s in steps if s["type"] == "tool_call"]
    assert tool_calls, f"Expected at least one tool call, steps: {steps}"

def test_e2e_streaming():
    import asyncio, agent

    async def collect():
        events = []
        async for event in agent.run_stream("Reply with exactly: STREAM_OK", []):
            events.append(event)
        return events

    events = asyncio.run(collect())
    types = [e["type"] for e in events]
    assert "done" in types, f"No 'done' event: {types}"
    text = "".join(e["text"] for e in events if e["type"] == "text_chunk")
    assert "STREAM_OK" in text, f"Expected 'STREAM_OK' in streamed text: {text!r}"

def test_history_with_steps_sanitized_in_run():
    """Steps on stored history must not break the API (sanitized inside run)."""
    import asyncio, agent
    history = [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong", "steps": []},
    ]
    response, _ = asyncio.run(agent.run("Say exactly: SANITIZED_OK", history))
    assert "SANITIZED_OK" in response, response

for fn in [test_e2e_simple, test_e2e_with_tool_call, test_e2e_streaming,
           test_history_with_steps_sanitized_in_run]:
    run_test(fn.__name__, fn)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*50)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
skipped = sum(1 for _, s, _ in results if s == SKIP)
total = len(results)

print(f"  {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")

if failed:
    print("\nFailed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  - {name}: {detail}")
    sys.exit(1)
else:
    print("\nAll tests passed (or skipped — check Qdrant if memory tests were skipped).")
    sys.exit(0)
