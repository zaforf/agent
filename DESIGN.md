# Agent System — Design Document

> **Authority**: This document is the canonical source of truth for expected system behavior. Anything the running system does that contradicts a statement here is a bug unless this document is explicitly updated first.

---

## 1. Overview

This is a personal AI assistant for Zafir, exposed as a web application. It runs a multi-turn agentic loop backed by a cascade of LLM providers, a persistent SQLite conversation store, and a long-term vector memory system. The assistant can fetch web pages, remember facts across sessions, and surgically edit its own system prompt.

**Core design priorities**
- Fast streaming responses with visible tool call steps in the UI
- Correct and complete context for the model on every turn (no silent data loss)
- Cheap-to-run: primary models are free-tier Google AI Studio; fallbacks are rate-limited free tiers
- Self-contained: no managed backends beyond Qdrant running locally

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Server | FastAPI (Python), served via uvicorn (gunicorn in prod) |
| LLM providers | Google AI Studio (Gemma 4), Cerebras, Groq |
| Primary API adapter | `gemini_client.py` — native Gemini REST API, currently Gemini API has a bug where recently generated keys results in 400 "Multiple authentication credentials received", if this is fixed we should revert to the system used for the other models to improve code reuse |
| Fallback API adapter | OpenAI Python SDK (`AsyncOpenAI`) |
| Conversation storage | SQLite (`data/history.db`) |
| Long-term memory | Mem0 + Qdrant (localhost:6333) |
| Memory embeddings | `BAAI/bge-base-en-v1.5` (local HuggingFace) |
| Memory LLM | Groq `llama-3.1-8b-instant` |
| Web UI | Vanilla JS + Marked (markdown) + KaTeX (LaTeX) |
| Streaming protocol | Server-Sent Events (SSE) |

---

## 3. Provider Chain

Providers are tried in order. On rate-limit (`RateLimitError`, `APIConnectionError`), the same provider is retried up to 3 times with exponential backoff (1s, 2s). On non-retryable errors (`APIError`, unexpected exceptions), the provider is skipped immediately and the next one is tried.

| Priority | Name | Model | Notes |
|---|---|---|---|
| 1 | `gemini-gemma4-31b` | `gemma-4-31b-it` | Primary — dense model, strong reasoning and tool use |
| 2 | `gemini-gemma4-26b` | `gemma-4-26b-a4b-it` | Secondary — MoE, 3.8B active params, very fast |
| 3 | `cerebras` | `qwen-3-235b-a22b-instruct-2507` | Fallback — 1M tokens/day free |
| 4 | `groq` | `llama-3.3-70b-versatile` | Last resort — no daily cap, always available |

A provider is only added to the active client list if its API key is present in the environment. Missing-key providers are silently skipped at startup.

If all providers are exhausted without success, the server raises `RuntimeError("All providers exhausted")`.

**Provider used** is returned in the API response and shown in the UI as a "via X" tag when the primary is not used.

### 3.1 Gemini Authentication

Google AI Studio keys with the `AQ.` prefix (current format) are incompatible with Bearer auth on the OpenAI-compatibility endpoint (`/v1beta/openai/`). The system uses the native Gemini REST API (`/v1beta/models/`) with `?key=` query-parameter authentication instead.

`gemini_client.py` is a full drop-in replacement for `AsyncOpenAI` that translates between OpenAI message format and the native Gemini REST API. It handles:
- `system` messages → `systemInstruction`
- `assistant` messages with `tool_calls` → `functionCall` parts
- `tool` messages → `functionResponse` parts (grouped under `user` role)
- Gemini `"thought": true` parts → wrapped in `<thinking>...</thinking>` tags for the stream stripper
- Streaming via `streamGenerateContent?alt=sse`

Cerebras and Groq use the standard `AsyncOpenAI` client pointed at their OpenAI-compatible endpoints.

---

## 4. Agentic Loop

Both `/chat` (non-streaming) and `/chat/stream` (SSE streaming) share the same logical loop. The loop runs up to `MAX_TOOL_ITERATIONS = 10` iterations before giving up with a "Reached max tool iterations" error.

### 4.1 Per-iteration flow

```
1. Call LLM with current messages (system prompt + full history + user message + any tool results so far)
2. AFTER the LLM response is received/streamed:
   - Resolve any pending history-summarization tasks from the PREVIOUS iteration
     (replacing old tool-result message content in-place with the summary)
3. If the LLM response contains tool calls:
   a. Execute each tool call sequentially
   b. Append tool results to messages (with full content)
   c. If a tool result exceeds 8,000 chars, start a background summarization task
      (does NOT block — runs concurrently during the next LLM call)
   d. Continue to next iteration
4. If the LLM response has no tool calls:
   - Strip thinking blocks from the response
   - If visible text is empty but raw content exists, issue a repair call
     (non-tool, single call asking the model to re-emit visible text only)
   - Return/emit the final response
```

**Critical invariant**: The model always receives the full, unsummarized tool output for the turn it is directly responding to. Summarization only replaces the content in the messages list *after* the model has responded to that tool — i.e., it only affects subsequent turns. This is achieved by resolving summaries at step 2, after the LLM call, not before.

### 4.2 Thinking blocks

Models may emit reasoning inside `<thought>`, `<think>`, `<thinking>`, `<redacted_reasoning>`, or `<redacted_thinking>` blocks. These are stripped before the response is shown to the user or stored in history.

In **non-streaming mode**, `_visible_after_think()` strips all closed reasoning blocks via regex.

In **streaming mode**, `_ThinkStripper` processes chunks in real-time:
- State machine: `scanning → buffering → passthrough`
- Buffers content inside thinking tags; passes through only visible content
- Emits `thinking_chars` events to the UI while buffering (drives the animated thinking indicator)
- Note: `_ThinkStripper` handles `thought|think|thinking` tags only; `redacted_reasoning` and `redacted_thinking` are only handled post-stream by `_visible_after_think`. Streaming models that emit those longer forms may pass them through raw. (Known limitation.)

Gemini's native thinking API uses `"thought": true` part metadata rather than inline tags. `gemini_client.py` wraps these in `<thinking>...</thinking>` tags so the stream stripper handles them uniformly.

### 4.3 Repair call

If the model's response is non-empty in raw form but produces no visible text after thinking-block stripping (e.g., the entire response was inside a reasoning block), the system sends a follow-up prompt (`_REPAIR_USER`) asking the model to re-emit just the user-visible answer. The repair uses non-streaming and disables tools.

See §11 for known edge cases in the streaming repair path (pinned by xfail tests in `tests/test_agent_loop.py`).

---

## 5. Tool System

Tools are registered in `tools/__init__.py`. Adding a new tool requires only creating a module with `SCHEMAS` and `FUNCTIONS` dicts and importing it there.

All tools are called synchronously. `fetch_url` is classified as a blocking sync tool (`_BLOCKING_SYNC_TOOLS`) and is run in a thread via `asyncio.to_thread()` to avoid blocking the event loop.

The model is instructed to use native API `tool_calls` only — no XML or fenced-code tool invocations.

### 5.1 Memory tools (`tools/memory.py`)

Backed by Mem0 + Qdrant. Qdrant host/port are read from `QDRANT_HOST` / `QDRANT_PORT` env vars (defaulting to `localhost:6333`); prod typically sets `QDRANT_HOST=qdrant` inside docker-compose. `docker-compose.yml` is gitignored because dev/prod topologies differ. Embeddings are computed locally using `BAAI/bge-base-en-v1.5` (768-dimensional). Mem0 uses Groq `llama-3.1-8b-instant` for memory extraction/processing.

All memories are stored under the single user ID `"user"`.

| Tool | Description |
|---|---|
| `remember(content, category)` | Store a durable fact. Categories: `user`, `preference`, `fact`, `project`. Only for facts worth recalling in a future conversation. |
| `recall(query)` | Semantic search over stored memories. Returns up to 5 results. |
| `list_memories()` | List all memories with IDs and categories. |
| `delete_memory(memory_id)` | Delete a specific memory by full ID. |

The system prompt instructs the model to use `recall()` before answering anything where past context is relevant, and to use `remember()` to build a map of the user's knowledge state (concepts mastered, depth of understanding, analogies that worked). The model is explicitly told never to store what was asked or what it answered.

If Qdrant is unreachable, memory tool calls fail with an exception caught by the tool executor, which returns the error string to the model.

### 5.2 Web fetch tool (`tools/fetch.py`)

`fetch_url(url, prompt, offset, raw)` fetches a URL and returns its content.

**Default (summarizer) mode** — when `prompt` is provided and `raw` is not set:
1. Fetch the URL with a browser-like User-Agent
2. Parse HTML using a custom `_TextExtractor` (strips `script`, `style`, `nav`, `header`, `footer`, `aside`, `noscript` tags; preserves body text with block-level newlines)
3. Truncate content to 128,000 characters if needed
4. Send the full text + the caller's prompt to `summarizer.summarize_gemma` — the shared Gemma 4 26B native-Gemini helper used by both this tool and `_summarize_for_history`
5. Return the summarizer's extracted/structured response (up to 8,192 output tokens)

The summarizer gives the model exactly what it asked for rather than a raw HTML dump. The prompt should describe what to extract (e.g., "list all albums in chronological order").

**Raw/paginated mode** — when `raw=True` or no prompt given:
- Returns up to 8,000 characters starting from `offset`
- Appends a pagination note: `[… N more chars — call fetch_url with offset=M to continue]`
- The model can call `fetch_url` with an increasing offset to walk through large documents

The model can always paginate regardless of whether a pagination note is visible. The note exists only as a convenience hint; it is preserved through history summarization specifically so the model doesn't lose track of where it left off.

Retry behavior: up to 4 retries on timeout or connection errors with exponential backoff; up to 3 retries on HTTP 429 (rate limit) with `Retry-After` header respect.

If the summarizer fails, the tool falls back to raw mode silently (logs a warning).

### 5.3 System prompt tools (`tools/self_modify.py`)

| Tool | Description |
|---|---|
| `get_system_prompt()` | Read the full current system prompt from `data/system_prompt.md` |
| `edit_system_prompt(new_prompt, reason)` | Overwrite the system prompt file |

The system prompt instructs the model to always call `get_system_prompt()` before `edit_system_prompt()`, to make surgical edits only (not rewrites), and to only modify when user feedback clearly requires a permanent behavior change. The file is read fresh on every turn (via `_build_system_prompt()`) so edits take effect immediately on the next call.

---

## 6. History & Context Management

### 6.1 Session model

Conversations are organized into sessions identified by a string `session_id`. The frontend generates a random session ID (stored in `localStorage`) on first visit. Sessions persist indefinitely until explicitly deleted.

### 6.2 In-memory cache

`main.py` maintains a per-process dict `_cache: dict[str, list[dict]]`. On first request for a session, history is loaded from SQLite into the cache. Subsequent requests read and write the cache directly. SQLite is the persistence layer; the cache is an optimization.

**On server restart**: cache is cleared. Next request reloads from SQLite. No data loss.

**Multi-instance note**: Each server process has its own cache. If multiple instances run behind a load balancer, their caches diverge. Not designed for multi-instance use.

### 6.3 What gets stored in history

After every completed turn, the full **turn messages** slice is stored. This includes:

1. The user message (`role: "user"`)
2. Any intermediate assistant messages with tool calls (`role: "assistant"`, `tool_calls: [...]`, `content: null`)
3. Tool result messages (`role: "tool"`, `name: <tool_name>`, `tool_call_id: <id>`, `content: <result or summary>`)
4. The final assistant message (`role: "assistant"`, `content: <visible response>`)

This full sequence is what gets fed back to the model on the next turn, giving it complete visibility into its own tool use history.

The **cache** is extended with `turn_messages` directly. SQLite stores `turn_messages` as a JSON blob on a single row per turn. The `content` column holds the user message text for session preview queries.

### 6.4 History sent to the model

Every LLM call receives:
```
[system prompt] + [sanitized history] + [user message] + [tool results so far this turn]
```

`_sanitize_history()` is whitelist-based — it keeps only API-safe keys (`role`, `content`, `tool_calls`, `name`, `tool_call_id`) and drops malformed messages. Anything else (stray UI fields, extensions) is silently discarded.

### 6.5 Tool result summarization

When a tool result exceeds **8,000 characters**, it is queued for background summarization. The summarizer:
- Receives: the last user message, the tool name, the tool arguments, and the full tool output
- Uses: **Gemma 4 26B** via `summarizer.summarize_gemma` — the same fast MoE model the `fetch_url` tool uses. Shared via the top-level `summarizer.py` module so there is exactly one summarizer implementation. The primary provider chain stays reserved for the agent loop.
- Returns: a compact summary preserving key facts, numbers, decisions, errors, and conclusions
- Is context-limited: the summarizer does NOT receive earlier conversation history, so summaries may be thin or generic if the goal was established several turns earlier (this is expected and noted in the system prompt)

**Non-blocking timing — DESIGN COMMITMENT.** Summarization MUST NEVER stall a turn. Concretely:

1. When a tool returns >8 000 chars, the agent starts the summary via `asyncio.create_task()` and keeps the **raw** content in the tool message.
2. At every iteration boundary and at the end of the turn, `_apply_finished_summaries` does a non-blocking poll (a single `await asyncio.sleep(0)` tick to let zero-latency stubs complete) and swaps in the summary only for tasks that are *already done*. In-flight tasks stay pending.
3. `agent.run()` returns `(response, provider, turn_messages, pending_summaries)`. `pending_summaries` is a list of `(message_dict, asyncio.Task)` pairs.
4. `main.py` appends the turn to SQLite with whatever content is currently in the dicts (raw, if the summary is still running) and then spawns `_finalize_summaries` as a fire-and-forget background task. When the summaries finish, that task mutates the in-memory `turn_messages` (the same dict objects cached per session) and calls `db.update_turn_messages(row_id, ...)` to overwrite the stored row.

The upshot: **the HTTP response returns the moment the model's final answer is ready**, regardless of how slow the summarizer is. The user never waits on history compaction. Subsequent turns see the summarized form as soon as the finalizer has run — typically within a second or two of the response, far before the user's next message.

**§4.1 ordering guard.** `_apply_finished_summaries` runs *after* `_call`, and the summary task itself never mutates the dict — it only returns a string. Together those two rules mean the summary can only be swapped in after the request carrying the raw content has already been sent to the API. Once the server has the request, a summary finishing during streamback is harmless. So even when the summarizer finishes before the primary model's response, the model still sees the raw tool output for the turn it is answering. Pinned by `test_raw_tool_content_preserved_even_when_summary_wins_race`.

**Pagination note preservation**: For `fetch_url` results, any `[… N more chars — call fetch_url with offset=M to continue]` note in the original output is extracted before summarization and re-appended to the summary. This ensures the model can continue paginating even after the raw content is compressed in history.

**Fallback**: If the summarizer raises or returns thinking-only (empty visible) output, the raw content is truncated to 8 000 characters instead. The DB row is never overwritten with an error — the raw-truncated form simply sticks.

### 6.6 Display history vs. LLM history

`db.get_history()` returns the full message sequence (including intermediate tool-call messages) for LLM context construction.

`db.get_display_history()` collapses each turn into a user message + an assistant message, with tool call/result pairs synthesized from the turn's `tool_calls` / `tool` messages and attached as a `steps` list on the assistant entry. This is the format the UI's `loadHistory()` expects and is served by `GET /sessions/{session_id}/history`.

Only `turn_messages` is stored; the UI-facing `steps` list is always derived on demand. This keeps one source of truth for what happened in a turn and means a future change to how a step is rendered never requires a schema migration.

---

## 7. SQLite Schema

Table: `messages` — one row per completed turn.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment row ID |
| `session_id` | TEXT | Session identifier |
| `role` | TEXT | Always `"user"` (legacy column; preserved for the preview query) |
| `content` | TEXT | User message text (used for session preview) |
| `turn_messages` | TEXT (JSON) | Full message sequence for the turn (user + intermediates + final assistant) |
| `ts` | INTEGER | Unix timestamp |

`init()` runs `CREATE TABLE IF NOT EXISTS` on startup.

---

## 8. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Non-streaming chat. Returns `{response, session_id, provider}`. Used by tests; the UI streams exclusively. |
| `POST` | `/chat/stream` | SSE streaming chat. Yields event objects (see §8.1). |
| `GET` | `/sessions` | List all sessions with preview text, message count, and last timestamp. |
| `GET` | `/sessions/{id}/history` | Display-friendly history for the UI (`get_display_history()`). |
| `DELETE` | `/sessions/{id}` | Delete a session from cache and SQLite. |
| `GET` | `/memories` | List all Mem0 memories. |
| `DELETE` | `/memories/{id}` | Delete a memory by ID. |
| `GET` | `/system-prompt` | Read current system prompt. |
| `PUT` | `/system-prompt` | Overwrite system prompt (used by UI editor). |
| `GET` | `/health` | Returns `{"status": "ok"}`. |
| `GET` | `/*` | Static files from `static/` (serves the web UI). |

### 8.1 SSE event types (`/chat/stream`)

| `type` | Fields | Description |
|---|---|---|
| `text_chunk` | `text: str` | Incremental visible text from the model |
| `thinking_chars` | `count: int` | Number of thinking chars buffered so far (drives indicator) |
| `tool_call` | `name: str`, `args: dict` | A tool is about to be called |
| `tool_result` | `name: str`, `result: str` | Tool execution completed |
| `done` | `provider: str`, `turn_messages: list` | Turn complete; `turn_messages` is the full history slice |
| `error` | `detail: str` | Unrecoverable error |

The frontend uses `turn_messages` from the `done` event to update the in-memory history cache (server-side). The client never manages history state directly.

---

## 9. Web UI

Single-page app (`static/index.html`). All state is managed client-side except conversation history (server-side in `_cache` / SQLite).

### 9.1 Layout

- **Left sidebar** (sessions panel): session list with preview text and timestamp; switch, create, delete sessions
- **Center** (chat): message feed + input bar
- **Right sidebar** (memories panel): list of all Mem0 memories with per-entry delete
- **Modal** (system prompt editor): full textarea edit + save via PUT `/system-prompt`

### 9.2 Message rendering

**User messages**: plain text, pre-wrap, dark bubble.

**Agent messages**: markdown rendered via Marked.js with `breaks: true, gfm: true`. LaTeX rendered via KaTeX `auto-render`. Each text segment between tool calls gets its own bubble.

**Tool steps**: collapsible rows between bubbles. Collapsed: shows direction tag (`>`/`<`), tool name, and a short meta hint (first arg value or result character count). Expanded: shows full args/result (capped at 600 chars display).

**Fallback tag**: if `provider` in the `done` event is not `"gemini-gemma4-31b"`, a small italic "via X" label appears above the response bubble.

**Thinking indicator**: animated dots above the bottom bar while the model is generating (1 to 2 to 3 to 2 to 1 dots repeatedly). Shows "received N chars" when thinking blocks are being buffered.

### 9.3 Streaming rendering

During streaming, text chunks are rendered with `renderStreamingMarkdown()` which tolerates incomplete code fences (closes unclosed ` ``` ` and `` ` `` blocks so the DOM stays well-formed mid-stream). On `done`, the final text is re-rendered with `renderMarkdown()` (standard Marked).

When a `tool_call` event arrives mid-stream:
1. The current text bubble is finalized (rendered with full markdown)
2. A new tool-call step row is appended
3. `agentMd` is reset to `null` so the next `text_chunk` creates a fresh bubble

This guarantees text → tool → text ordering is reflected in the DOM regardless of what arrives when.

### 9.4 History loading on page load / session switch

Calls `GET /sessions/{id}/history` which returns the display-friendly format (`get_display_history()`). For each message:
- `role: "user"` → user bubble
- `role: "assistant"` → tool step rows (from `msg.steps`, synthesized server-side from the turn's tool_calls/tool messages) followed by agent bubble

### 9.5 Session ID persistence

Session ID is stored in `localStorage` as `"sid"`. Survives page reloads and browser restarts. Cleared only by `newSession()` or explicit deletion.

---

## 10. System Prompt

The system prompt is stored in `data/system_prompt.md` and read on every LLM call via `_build_system_prompt()` (so edits take effect immediately). It is prepended to the tool documentation section generated from `TOOL_SCHEMAS`.

### 10.1 Current behavioral directives

**Persona**: Personal AI assistant for Zafir. Highly efficient and concise. No hedging, no over-explaining, no trailing summaries.

**Preferences**:
- Concise technical answers by default
- No padding or question restatement
- Visible answer must always appear outside reasoning blocks
- Use native API `tool_calls` only (no XML or fenced-code tool invocations)

**Source grounding**:
- If a URL is provided, prioritize `fetch_url` for requests implying deep analysis or source-specific perspective
- Reserve internal knowledge for trivial facts or when the URL is clearly supplementary

**Grounding & tool trust**:
- Training cutoff is January 2025; anything after that is unknown — tool results are almost certainly more accurate
- Do not flag tool results as suspicious just because they conflict with internal knowledge; the more likely explanation is that internal knowledge is outdated

**Tool history trust**:
- History summaries (`[history summary of tool_name]`) are produced by a capable model with access to the full original output — treat them as accurate
- The current turn's tool result is always passed in full; summarization only affects older turns
- Sparse summaries are a compression artifact (the summarizer only receives the last user message and tool args, not full history) — a sparse summary is not evidence the tool was unhelpful or that a past response came from internal knowledge
- Pagination: the model can always call `fetch_url` again with an offset; when a pagination note survived summarization it will be visible

**Memory discipline**:
- Use `recall()` before answering anything where past context or knowledge level is relevant
- Store: concepts mastered, depth of understanding, successful analogies
- Never store: what was asked, what was answered, trivial temporary context

**Self-modification rules**:
- Always call `get_system_prompt()` before `edit_system_prompt()`
- Surgical edits only — preserve everything else
- Only modify for permanent behavior changes, not one-off requests

---

## 11. Known Limitations

This section lists remaining accepted trade-offs for this single-user deployment. Previously-flagged behavioral bugs (original §11.1 streaming repair and §11.4 summarizer model choice) have been fixed and are covered by regular green tests, not `xfail`s. There are no outstanding `xfail` bugs.

### 11.1 Streaming repair (resolved)

The streaming repair gate now keys on visible content (`visible_parts`), not raw content, so a stream consisting entirely of `<thinking>...</thinking>` correctly triggers the repair pass. The repair call runs against a scratch message list so the scaffold (`_REPAIR_USER` + the empty-visible assistant placeholder) does NOT leak into stored `turn_messages`. Verified by `tests/test_agent_loop.py::test_streaming_repair_triggers_on_thinking_only`.

### 11.2 Summarizer uses Gemma 26B (resolved)

`_summarize_for_history()` now uses Gemma 4 26B via the shared `summarizer.summarize_gemma` helper (~10× faster than the dense primary, fine for compact structured summaries). See §6.5. Verified by `tests/test_agent_helpers.py::test_summarize_for_history_uses_gemma_26b`.

---

## 12. Test suite as source of truth

Tests live in `tests/` and are the operational form of this document. Any discrepancy between the test assertions and the running code is a bug.

- **`pytest`** (default): ~100 fast, hermetic tests. No network, no Qdrant, no real DB. Mocks the provider chain via scripted `_FakeCompletions` on `agent._clients` and the tool registry via `monkeypatch` on `agent.TOOL_FUNCTIONS`.
- **`pytest -m live`**: end-to-end tests against the real provider chain (simple response, history round-trip, multi-chunk streaming, tool-call trace, fetch+summarize integration). Kept small to respect free-tier rate limits.
- **Markers**: `live` (real LLM), `net` (real HTTP). Both deselected by default via `pyproject.toml`.

The most important invariant tests — the ones that answer "is the agent doing what the design document says?" — live in `test_agent_loop.py`: tool results flowing back into the next LLM call, the summarization-timing invariant (§4.1), the non-blocking summarization contract (§6.5), repair-on-empty-visible (§4.3), provider fallback, and the streaming event contract. The suite currently has **zero `xfail` tests** — every known behavioral commitment is pinned by a green assertion.

### 12.1 Fall-through coverage

Graceful degradation is tested explicitly — the agent must never crash when an external dependency misbehaves:

| Failure mode | Expected behavior | Test |
|---|---|---|
| Provider raises retryable error (RateLimit, APIConnection) | Retry 3× with backoff, then move to next provider in chain | `test_provider_fallback_on_retryable_error` |
| Provider raises `APIError` | Skip immediately to next provider (no retries) | `test_api_error_skips_provider_immediately` |
| Every provider in the chain fails | `RuntimeError("All providers exhausted")` — `main.py` converts to HTTP 500 | `test_all_providers_exhausted_raises_cleanly` |
| Tool callable raises an exception | Catch and inject `"Error in {tool}: {msg}"` as the tool message; loop continues so the model can react | `test_tool_exception_returned_as_error_string` |
| Model hallucinates an unknown tool name | Inject `"Unknown tool: {name}"` as the tool message; loop continues | `test_unknown_tool_name_returns_error_string` |
| `fetch_url` summarizer LLM fails | Fall back to raw paginated text (DESIGN §5.2) | `test_summarizer_failure_falls_back_to_raw` |
| History summarizer LLM fails | Truncate raw output to 8 000 chars and store that (DESIGN §6.5) | `test_summarize_for_history_truncates_when_llm_raises` |
| History summarizer returns thinking-only (empty visible) | Same fallback — truncate raw to 8 000 chars | `test_summarize_for_history_truncates_when_summary_is_empty` |
| Model produces only thinking tags (empty visible) | Repair call with `tools=None` re-asks for a user-facing answer (DESIGN §4.3) | `test_repair_call_on_empty_visible` |
| Model produces only thinking tags during streaming | Same — repair fires (§11.1 fix); scaffold stays out of history | `test_streaming_repair_triggers_on_thinking_only` |
| History summarizer still running when turn ends | `agent.run()` returns immediately; `main.py` drains the task in the background and patches the stored row (DESIGN §6.5) | `test_history_summarization_is_non_blocking`, `test_chat_non_blocking_summary_patches_db_row` |
| Summarizer finishes *before* the LLM responding to that tool result | Summary is NOT swapped in until after the LLM call returns — the model always sees raw content for the turn it is answering (§4.1, §6.5) | `test_raw_tool_content_preserved_even_when_summary_wins_race` |
