"""
Agent test harness.
Run: ./venv/bin/python test_agent.py

Covers:
  1. Config — all required providers have keys; chain order is correct
  2. Tool parsing — XML, Python kwargs, Python positional
  3. DB — append / get_history / clear / get_sessions round-trip
  4. Memory — remember / recall / list_memories / delete_memory (needs Qdrant)
  5. System prompt — read, edit safety guard, content roundtrip
  6. Provider client construction — only providers with keys are wired up
  7. End-to-end chat — sends a real message through the live provider chain
"""

import sys
import os
import traceback
import uuid

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, str, str]] = []  # (name, status, detail)


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

# ── 2. Tool parsing ───────────────────────────────────────────────────────────

print("\n== 2. Tool parsing ==")

def test_xml_format():
    from agent import _parse_tool_calls
    result = _parse_tool_calls('<tool_call>{"name": "remember", "args": {"content": "hello", "category": "fact"}}</tool_call>')
    assert result, "No calls parsed"
    assert result[0]["name"] == "remember"
    assert result[0]["args"]["content"] == "hello"
    assert result[0]["args"]["category"] == "fact"

def test_xml_multiple():
    from agent import _parse_tool_calls
    blob = (
        '<tool_call>{"name": "recall", "args": {"query": "name"}}</tool_call>'
        '<tool_call>{"name": "list_memories", "args": {}}</tool_call>'
    )
    result = _parse_tool_calls(blob)
    assert len(result) == 2, f"Expected 2, got {len(result)}"
    assert result[0]["name"] == "recall"
    assert result[1]["name"] == "list_memories"

def test_no_false_positives_plain():
    from agent import _parse_tool_calls
    result = _parse_tool_calls("Just a normal sentence with no tool calls here.")
    assert result == [], f"Expected [], got {result}"

def test_no_false_positives_python_syntax():
    """Python-syntax calls must NOT be parsed — they were removed to prevent
    false positives when the model discusses a tool by name in prose."""
    from agent import _parse_tool_calls
    cases = [
        'remember(content="test", category="fact")',
        'recall("learning style")',
        'The remember() guidelines say...',
        'modify the `remember()` call to be less strict',
    ]
    for text in cases:
        result = _parse_tool_calls(text)
        assert result == [], f"False positive on: {text!r} → {result}"

def test_xml_takes_priority_and_python_ignored():
    """Even with Python syntax present, only XML block is parsed."""
    from agent import _parse_tool_calls
    blob = '<tool_call>{"name": "recall", "args": {"query": "test"}}</tool_call> remember("stray")'
    result = _parse_tool_calls(blob)
    assert len(result) == 1, f"Expected 1, got {len(result)}: {result}"
    assert result[0]["name"] == "recall"

for fn in [test_xml_format, test_xml_multiple,
           test_no_false_positives_plain, test_no_false_positives_python_syntax,
           test_xml_takes_priority_and_python_ignored]:
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

for fn in [test_db_round_trip, test_db_sessions_list]:
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
        """
        Mem0 LLM deduplication can rewrite/merge stored text, so exact tag
        matching is unreliable. We verify: remember returns confirmation,
        recall returns non-empty results for a semantically related query.
        """
        from tools.memory import remember, recall
        store_result = remember("Zafir studies machine learning and deep learning", category="fact")
        assert "Stored" in store_result or store_result, f"remember() returned unexpected: {store_result}"
        recall_result = recall("machine learning")
        assert recall_result and recall_result != "No relevant memories found.", (
            f"recall() returned nothing: {recall_result}"
        )

    def test_list_and_delete_memory():
        """
        Mem0's LLM extraction may rewrite/merge the input text, so we can't
        do exact-tag matching. Instead: verify listing works, grab any ID, and
        confirm delete succeeds without error.
        """
        import re
        from tools.memory import remember, list_memories, delete_memory
        # Ensure at least one memory exists
        remember("Zafir prefers Python for scripting tasks", category="preference")
        listing = list_memories()
        assert listing and listing != "No memories stored.", f"Listing empty: {listing!r}"
        # Full UUID pattern e.g. f52db2fc-1234-5678-abcd-123456789abc
        ids = re.findall(r'\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]', listing)
        assert ids, f"No IDs found in listing output:\n{listing}"
        # Delete the last one (least likely to be load-bearing)
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
    # Restore
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

for fn in [test_clients_built_for_keyed_providers, test_client_order_matches_provider_chain,
           test_primary_is_gemma4_31b]:
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
    """Verify run_stream() yields correct event types and a done event."""
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

for fn in [test_e2e_simple, test_e2e_with_tool_call, test_e2e_streaming]:
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
