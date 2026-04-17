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

**Known issue**: In the streaming repair path, the repair scaffold messages (`_REPAIR_USER` + the empty assistant placeholder) are included in `turn_messages` and thus written to history. In the non-streaming path, the repair response is not appended to `messages` at all, so `turn_messages` ends with the `_REPAIR_USER` message rather than the final assistant text. Both cases represent minor history corruption in a rare edge case.

---

## 5. Tool System

Tools are registered in `tools/__init__.py`. Adding a new tool requires only creating a module with `SCHEMAS` and `FUNCTIONS` dicts and importing it there.

All tools are called synchronously. `fetch_url` is classified as a blocking sync tool (`_BLOCKING_SYNC_TOOLS`) and is run in a thread via `asyncio.to_thread()` to avoid blocking the event loop.

The model is instructed to use native API `tool_calls` only — no XML or fenced-code tool invocations.

### 5.1 Memory tools (`tools/memory.py`)

Backed by Mem0 + Qdrant. Qdrant runs locally at `localhost:6333` (issue: probably best to make QDRANT_HOST an env variable, as in production its actually qdrant:6333. related, we want to gitignore docker-compose.yml and rm the commited version, keep it locally, as prod also has different docker-compose.yml). Embeddings are computed locally using `BAAI/bge-base-en-v1.5` (768-dimensional). Mem0 uses Groq `llama-3.1-8b-instant` for memory extraction/processing.

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
4. Send the full text + the caller's prompt to the Gemini summarizer model (`gemma-4-26b-a4b-it`) via the native Gemini API with `?key=` auth
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

The **cache** is extended with `turn_messages` directly. SQLite stores `turn_messages` as a JSON blob in the `turn_messages` column of the user row. The `role`/`content` columns hold the user message text for session preview queries.

### 6.4 History sent to the model

Every LLM call receives:
```
[system prompt] + [sanitized history] + [user message] + [tool results so far this turn]
```

`_sanitize_history()` strips UI-only fields (`steps`, etc.) and drops malformed messages before sending. Tool messages from history pass through sanitization intact (including `name` and `tool_call_id` fields needed by Gemini).

### 6.5 Tool result summarization

When a tool result exceeds **8,000 characters**, it is queued for background summarization. The summarizer:
- Receives: the last user message, the tool name, the tool arguments, and the full tool output
- Uses: the first available provider from `_clients` (same chain as the main model)
- Returns: a compact summary preserving key facts, numbers, decisions, errors, and conclusions
- Is context-limited: the summarizer does NOT receive earlier conversation history, so summaries may be thin or generic if the goal was established several turns earlier (this is expected and noted in the system prompt)

**Timing**: The summarization task runs concurrently with the next LLM call (using `asyncio.create_task`). It is awaited and its result applied to the messages list *after* that LLM call completes — i.e., the model that directly responds to a tool always sees the full output; the summary only replaces content for subsequent turns.

**Pagination note preservation**: For `fetch_url` results, any `[… N more chars — call fetch_url with offset=M to continue]` note in the original output is extracted before summarization and re-appended to the summary. This ensures the model can continue paginating even after the raw content is compressed in history.

**Fallback**: If summarization fails, the raw content is truncated to 8,000 characters instead.

### 6.6 Display history vs. LLM history

`db.get_history()` returns the full message sequence (including intermediate tool-call messages) for LLM context construction.

`db.get_display_history()` collapses each turn into a user message + an assistant message, with tool call/result pairs attached as `steps` on the assistant. This is the format the UI's `loadHistory()` expects and is served by the `/sessions/{session_id}/history` endpoint.

**Legacy format**: Rows created before the `turn_messages` column was added (old `append()` calls) have no `turn_messages` value. Both `get_history()` and `get_display_history()` handle these by falling back to the simple `role`/`content` values stored in the row. issue: we can get rid of this, theres no critical old chats to maintain.

---

## 7. SQLite Schema

Table: `messages`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment row ID |
| `session_id` | TEXT | Session identifier |
| `role` | TEXT | Always `"user"` for new-format rows |
| `content` | TEXT | User message text (used for session preview) |
| `steps` | TEXT (JSON) | Tool call/result events for display (list of step dicts) |
| `turn_messages` | TEXT (JSON) | Full message sequence for the turn (user + intermediates + final assistant) |
| `ts` | INTEGER | Unix timestamp |

`init()` runs migrations on startup to add missing columns to existing databases.

---

## 8. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Non-streaming chat. Returns `{response, session_id, steps, provider}`. |
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
- `role: "assistant"` → tool step rows (from `msg.steps`) followed by agent bubble

The legacy `steps` field on assistant messages carries tool call/result data from both old-format rows and new-format rows (new rows have steps reconstructed by `get_display_history()`). issue: get rid of this legacy handling, again no previous chats are important

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

## 11. Known Limitations & Noted Potential Bugs

### 11.1 Streaming repair path corrupts history

When the model's streaming response produces no visible text (entire output inside thinking blocks), a repair call is issued. In the streaming path, the empty assistant placeholder and `_REPAIR_USER` scaffold messages are included in `turn_messages` and written to SQLite. In the non-streaming path, the repair response is not appended to `messages`, so `turn_messages` ends with `_REPAIR_USER` rather than the final response. Both cases leave the model's history slightly malformed in a rare edge case.

### 11.2 `_ThinkStripper` does not handle `redacted_reasoning` / `redacted_thinking`

The streaming stripper only recognizes `thought`, `think`, and `thinking` open tags. If a model emits `<redacted_reasoning>` or `<redacted_thinking>` tags during streaming (as opposed to non-streaming), they pass through to the user unstripped. Post-stream, `_visible_after_think()` handles all five forms, so they are stripped before storage. This may be a non issue as they may not be emitted by the models we use, research required.

### 11.3 Session message count reflects rows, not messages

`get_sessions()` returns a `count` that is the number of database rows for the session. New-format turns produce one row each; legacy turns produce two (user + assistant). The UI shows this count as "N msgs" which is now turn-count for new sessions but message-count for legacy sessions.

### 11.4 Memory system is single-user

All memories use the hardcoded user ID `"user"`. There is no multi-user separation. Acceptable for a personal assistant but not suitable for shared deployments. This is fine, there will only ever be a single user.

### 11.5 Summarizer uses the primary model

`_summarize_for_history()` calls `_call(messages, use_tools=False)` which walks the full provider chain starting from the primary (Gemma 4 31B). Using the primary model for summarization is more expensive than necessary; a lighter model could be used. Not a correctness bug, but a resource efficiency note. this is probably fine even though there is a mismatch (31b used here, 26b used for fetch summarization, but thats because fetch summarization may be up to 128k characters long, and 26b is comparatively faster as a MoE).

### 11.6 `any(raw_parts)` check in streaming repair is imprecise

The condition `if not any(raw_parts)` checks whether the raw content list contains any truthy value. A model that outputs only `<thinking>...</thinking>` with no visible text would still populate `raw_parts` (with the tag content), causing `any(raw_parts)` to be True and taking the non-repair path. The assistant message stored in history would then have `visible_content = ""`. The check should arguably be `if not any(visible_parts)`.

### 11.7 No concurrency control on system prompt file

`edit_system_prompt()` does a direct file write with no locking. Concurrent requests editing the system prompt would produce a race condition. Acceptable for single-user use.

### 11.8 In-memory cache is per-process

Multiple server processes (e.g., uvicorn with multiple workers) each maintain their own `_cache`. History written by one worker is not visible to another until it reloads from SQLite on the next cache miss. Running with `--workers 1` (the default for development) avoids this. deployment runs with 2 workers now, if this is an uissue
