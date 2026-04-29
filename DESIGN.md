# Agent System — Design Document

> **Authority**: This document is the canonical source of truth for expected system behavior. Anything the running system does that contradicts a statement here is a bug unless this document is explicitly updated first.

---

## 1. Overview

This is a personal AI assistant for Zafir, exposed as a web application. It runs a multi-turn agentic loop backed by a cascade of LLM providers, a persistent SQLite conversation store, and a long-term vector memory system. The assistant can fetch web pages, remember facts across sessions, and surgically edit its own system prompt.

**Core design priorities**
- Fast streaming responses with visible tool call steps in the UI
- Correct and complete context for the model on every turn (no silent data loss)
- Cheap-to-run: primary models are free-tier Google AI Studio; fallbacks are rate-limited free tiers
- Self-contained: no paid managed backends beyond **Qdrant** (typically local or your own container) and **API usage** for models/embeddings

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Server | FastAPI (Python), served via uvicorn (gunicorn in prod) |
| LLM providers | Google AI Studio (Gemma 4), Cerebras, Groq |
| Gemini keys | **Tier-1** (`GEMINI_API_KEY`) — agent loop / `_clients`. **Free tier** (`GEMINI_API_KEY_FREE`) — Gemma 26B only; resolved at import to `config.GEMINI_API_KEY_FREE_RESOLVED` (fallback: `GEMINI_API_KEY`, one warning if free unset). |
| API adapter (all providers) | OpenAI Python SDK (`AsyncOpenAI`); Gemini uses Google's OpenAI-compatibility endpoint (`/v1beta/openai/`) |
| Conversation storage | SQLite (`data/history.db`) |
| Long-term memory | Mem0 + Qdrant (localhost:6333) |
| Memory embeddings | Google **Gemini Embedding** (`models/gemini-embedding-001` via `google-genai`; same `GEMINI_API_KEY` as chat). Dimensions default **768** (`GEMINI_EMBEDDING_DIMS`). No local `sentence-transformers` — suitable for small VPS RAM. |
| Memory LLM | Gemini API via mem0 (`MEM0_LLM_MODEL`, default `gemma-4-26b-a4b-it` — same `GEMINI_API_KEY` as chat; avoids Groq TPM limits on large extraction prompts) |
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

The **Gemma 26B summarizer** (`summarizer.summarize_gemma`) is not part of that chain: HTTP calls use `config.GEMINI_API_KEY_FREE_RESOLVED` (from `GEMINI_API_KEY_FREE` or fallback — see table above). That keeps summarization on a separate key/quota when configured.

If all providers are exhausted without success, the server raises `RuntimeError("All providers exhausted")`.

**Provider used** is returned in the API response and shown in the UI as a "via X" tag when the primary is not used.

### 3.1 Gemini Authentication

All Gemini calls use Google's OpenAI-compatibility endpoint (`/v1beta/openai/`) via `AsyncOpenAI` with the API key as a Bearer token — same pattern as Cerebras and Groq. AI Studio keys (`AIza…`) authenticate this way.

> Historical note: an earlier `gemini_client.py` translated to/from the native `/v1beta/models/` REST API (`?key=` param) when AI Studio's `AQ.`-prefix keys could not authenticate via Bearer. That adapter has been removed; if a future key format breaks compat again, reintroduce a thin native client behind a flag rather than silently regressing.

---

## 4. Agentic Loop

Both `/chat` (non-streaming) and `/chat/stream` (SSE streaming) share the same logical loop. The loop runs up to `MAX_TOOL_ITERATIONS = 30` iterations before giving up with a "Reached max tool iterations" error.

### 4.1 Per-iteration flow

```
1. Call LLM with current messages (system prompt + full history + user message + any tool results so far)
2. If the LLM response contains tool calls:
   a. Append the assistant message (with tool_calls)
   b. Execute each tool call sequentially
   c. Append tool results to messages (with full content)
   d. If a tool result exceeds 8,000 chars, start a background summarization task
       (does NOT block — runs concurrently during later LLM calls in this turn)
   e. Continue to next iteration (go to step 1)
3. If the LLM response has no tool calls:
   a. Strip thinking blocks / streaming finalize; repair if needed
   b. Resolve any finished history-summarization tasks (in-place swap to
      `[history summary of …]` for completed tasks only) — see §6.5
   c. Return/emit the final response
4. If max tool iterations exceeded: resolve finished summaries, then return error
```

**Critical invariant (precise):** For each tool-result message, **every** LLM call in the **same user turn** that runs **after** that message was appended sees the **full, unsummarized** text. In-place summary swaps run **only** when the turn is finishing (no more tool rounds in step 3, or step 4) — **not** between tool iterations. Therefore:

- **Same assistant `tool_calls` batch:** Unchanged — all results from that batch are full on the next `_call`.
- **Chained tools across iterations (A then B):** The `_call` after B is appended still sees **A and B both raw**. After the final text reply, finished summaries may replace large tool rows before persist; unfinished tasks stay pending for `main.py`'s finalizer.
- **Across user turns:** Replay may show `[history summary of …]` for prior turns as before.

The ordering guarantee that must never regress: `_apply_finished_summaries` must **not** run between step 1 and the next tool execution round — only at turn end (or from `main.py` after persist). The model must never receive a summary **instead of** raw text for a tool result on a `_call` that happens **before** the turn's final no-tools reply. Pinned by `test_raw_tool_content_preserved_even_when_summary_wins_race` and `test_multi_iteration_tools_prior_results_stay_raw_until_turn_end`.

### 4.2 Thinking blocks

Models may emit reasoning inside `<thought>`, `<think>`, `<thinking>`, `<redacted_reasoning>`, or `<redacted_thinking>` blocks. These are stripped before the response is shown to the user or stored in history.

In **non-streaming mode**, `_visible_after_think()` strips all closed reasoning blocks via regex.

In **streaming mode**, `_ThinkStripper` processes chunks in real-time:
- State machine: `scanning → buffering → scanning` (loops — closing tag returns to `scanning` so multiple interleaved thinking/output/thinking cycles are all handled correctly)
- Buffers content inside thinking tags; passes through only visible content
- Emits `thinking_chars` events to the UI while buffering (drives the animated thinking indicator). The UI throttles DOM writes to at most one per 100ms via `performance.now()` so rapid events cannot starve the renderer.
- Emits `thinking_done` when the `buffering → scanning` transition fires (i.e., the closing tag is received). This is necessary because the model may not emit any visible text token immediately after closing its thinking block — without `thinking_done` the counter would freeze at its last value during the generation gap. The UI reacts by clearing the count label while keeping the animated dots visible, so the indicator remains accurate until the first `text_chunk` arrives and `_hideThinking()` is called.
- Note: `_ThinkStripper` handles `thought|think|thinking` tags only; `redacted_reasoning` and `redacted_thinking` are only handled post-stream by `_visible_after_think`. Streaming models that emit those longer forms may pass them through raw. (Known limitation.)

Gemini's reasoning is exposed via the OpenAI-compat endpoint as inline `<thinking>` (or equivalent) text in the response, so the same stream stripper handles all providers uniformly.

### 4.3 Repair call

If the model's response is non-empty in raw form but produces no visible text after thinking-block stripping (e.g., the entire response was inside a reasoning block), the system sends a follow-up prompt (`_REPAIR_USER`) asking the model to re-emit just the user-visible answer. The repair uses non-streaming and disables tools.

See §11 for the repaired streaming edge case and the summarizer model choice, pinned by `tests/test_agent_loop.py::test_streaming_repair_triggers_on_thinking_only` and `tests/test_agent_helpers.py::test_summarize_for_history_uses_gemma_26b`.

---

## 5. Tool System

Tools are registered in `tools/__init__.py`. Adding a new tool requires only creating a module with `SCHEMAS` and `FUNCTIONS` dicts and importing it there.

All tools are called synchronously. `fetch_url`, `web_search`, and the Mem0 tools (`remember`, `recall`, `list_memories`, `delete_memory`) are classified as blocking sync tools (`_BLOCKING_SYNC_TOOLS`) and are run in a thread via `asyncio.to_thread()` to avoid blocking the event loop.

The model is instructed to use native API `tool_calls` only — no XML or fenced-code tool invocations.

### 5.1 Memory tools (`tools/memory.py`)

Backed by Mem0 + Qdrant. Qdrant host/port are read from `QDRANT_HOST` / `QDRANT_PORT` env vars (defaulting to `localhost:6333`); prod typically sets `QDRANT_HOST=qdrant` inside docker-compose. `docker-compose.yml` is gitignored because dev/prod topologies differ. Embeddings use the **Gemini Embedding API** (`GEMINI_API_KEY`, model `GEMINI_EMBEDDING_MODEL` defaulting to `models/gemini-embedding-001`, `GEMINI_EMBEDDING_DIMS` default 768). Vectors are stored under collection `MEM0_QDRANT_COLLECTION` (default `agent_memories_gemini` — new name so a prior local 768-d HuggingFace index is not reused). Mem0 uses the **Gemini** LLM provider (`MEM0_LLM_MODEL`, default **Gemma 4 26B MoE** `gemma-4-26b-a4b-it`) for memory extraction/processing — same `GEMINI_API_KEY` as the agent. This avoids Groq free-tier **tokens-per-minute** failures when the extraction prompt is large. Override with `MEM0_LLM_MODEL` (e.g. `gemma-4-31b-it`) if needed. Memory tools run in a worker thread (`asyncio.to_thread`) like `fetch_url` / `web_search` so the async event loop is not blocked during embedding or Qdrant I/O.

All memories are stored under the single user ID `"user"`.

| Tool | Description |
|---|---|
| `remember(content, category)` | Store a durable fact. Categories: `user`, `preference`, `fact`, `project`. Prefer **atomic memories** (one distinct fact/preference per call) unless points are inseparable, to improve retrieval precision. Only for facts worth recalling in a future conversation. Implementation uses Mem0 **`infer=False`** so the exact string is embedded and written to Qdrant; Mem0’s default **`infer=True`** path uses an LLM to extract facts and can persist nothing if extraction fails, which made `recall` / `list_memories` look empty despite a “Stored” reply. |
| `recall(query)` | Semantic search over stored memories. Returns up to 5 results. |
| `list_memories()` | List all memories with IDs and categories. |
| `delete_memory(memory_id)` | Delete a specific memory by full ID. |

The system prompt instructs the model to use `recall()` before answering anything where past context is relevant, and to use `remember()` to build a map of the user's knowledge state (concepts mastered, depth of understanding, analogies that worked). The model is explicitly told never to store what was asked or what it answered.

If Qdrant is unreachable, memory tool calls fail with an exception caught by the tool executor, which returns the error string to the model.

**Dependency:** `mem0ai` is pinned to **2.x** (`requirements.txt`: `mem0ai[nlp]>=2.0.0,<3`). Version 2 uses `Memory.search(..., filters={"user_id": "..."}, top_k=...)` and `get_all(filters={...})`; version 1 used `user_id=` / `limit=` instead — mixing code written for one major with the other produces runtime errors. The `[nlp]` extra installs spaCy so mem0’s optional lemmatization path does not warn on every import.

### 5.2 Web tools (`tools/web.py`)

Two tools live here: **`fetch_url`** for retrieving and reading a known URL, and **`web_search`** for discovering URLs when the model has none.

#### 5.2.1 `fetch_url(url, prompt, offset, raw)`

Fetches a URL and returns its content.

**Default (summarizer) mode** — when `prompt` is provided and `raw` is not set:
1. Fetch the URL with a browser-like User-Agent
2. Parse HTML using a custom `_TextExtractor` (strips `script`, `style`, `nav`, `header`, `footer`, `aside`, `noscript` tags; preserves body text with block-level newlines)
3. Truncate content to 128,000 characters if needed
4. Send the full text + the caller's prompt to `summarizer.summarize_gemma` — the shared Gemma 4 26B native-Gemini helper used by both this tool and `_summarize_for_history` (uses `config.GEMINI_API_KEY_FREE_RESOLVED` — see §2)
5. Return the summarizer's extracted/structured response (up to 8,192 output tokens)

The summarizer gives the model exactly what it asked for rather than a raw HTML dump. The prompt should describe what to extract (e.g., "list all albums in chronological order").

**Raw/paginated mode** — when `raw=True` or no prompt given:
- Returns up to 8,000 characters starting from `offset`
- Appends a pagination note: `[… N more chars — call fetch_url with offset=M to continue]`
- The model can call `fetch_url` with an increasing offset to walk through large documents

**Verification pitfall:** prompt mode exposes the summarizer to up to 128,000 characters of extracted text; raw mode exposes only 8,000 characters per call from a given `offset`. The first raw chunk is often intro/nav/infobox. The system prompt therefore instructs the agent not to treat "fact X missing from the first raw window" as evidence that prompt-mode output was hallucinated — the likelier explanation is that X appears deeper in the page and requires pagination or a second prompt-mode extraction.

The model can always paginate regardless of whether a pagination note is visible. The note exists only as a convenience hint; it is preserved through history summarization specifically so the model doesn't lose track of where it left off.

Retry behavior: up to 4 retries on timeout or connection errors with exponential backoff; up to 3 retries on HTTP 429 (rate limit) with `Retry-After` header respect.

If the summarizer fails, the tool falls back to raw mode silently (logs a warning).

#### 5.2.2 `web_search(query, max_results=5)`

Returns a numbered markdown list of `title — url` plus a short snippet (≤ 240 chars), backed by the **[Brave Search API](https://api.search.brave.com/)** (free tier; bring-your-own key via `BRAVE_SEARCH_API_KEY` in `.env`). Additive to `fetch_url` — the model uses search to find URLs and then `fetch_url` for depth.

- `max_results` is clamped to `[1, 10]` (default 5). Snippets and titles have HTML highlight tags stripped.
- If `BRAVE_SEARCH_API_KEY` is unset, the tool returns the stable error string `"Error: web_search disabled — set BRAVE_SEARCH_API_KEY in .env"` so the model can react. HTTP / timeout failures also return short `Error: …` strings.
- Single endpoint (`/res/v1/web/search`), 10 s timeout, no retry — Brave's free tier is rate-limited and one failure is enough signal for the model to switch strategies.

### 5.3 Shell tool (`tools/shell.py`)

| Tool | Description |
|---|---|
| `shell_exec(command, timeout)` | Execute a shell command in a persistent bash session; returns combined stdout+stderr |

**Architecture — single long-lived bash process:**

A single `subprocess.Popen(bash)` is started on first use and reused for all subsequent calls. All calls are serialized via `threading.Lock`. State — environment variables, working directory, installed packages — carries over between calls. stderr is merged into stdout so the model sees error messages inline.

**Workspace:**

The shell starts in `WORKSPACE` (resolved from the `AGENT_WORKSPACE` env var; default `<project-root>/workspace`). The directory is created automatically on first use. In production, bind-mount it to a host directory so files survive container rebuilds; the shell process itself may be lost on redeploy but files in the workspace are unaffected.

**Output and exit codes:**

- Non-zero exit codes are appended as `(exit code N)` after the output.
- Commands with no output return the string `"(no output)"`.
- Output exceeding 1 MB triggers a hard cap: the shell is restarted and a truncation notice is appended. Normal outputs that exceed 8 000 chars are summarized by the existing agent-loop history summarizer (§6.5) before storage, just like all other tools.

**Timeout:**

Default timeout is 30 seconds (overridable via `AGENT_SHELL_TIMEOUT` env var or per-call `timeout` parameter). On timeout the shell is killed and restarted for the next call. The system prompt instructs the model to increase `timeout` for long-running tasks (package installs, builds) and to use non-interactive flags for commands that would otherwise wait for user input.

**Shell restart:**

If the shell process dies unexpectedly (e.g., OOM kill), `_ensure_shell()` detects it on the next call (via `Popen.poll()`) and starts a fresh shell. The new shell always starts in `WORKSPACE`.

**Sentinel mechanism:**

Every command is appended with `__rc=$?; echo "<sentinel>:$__rc"` on a separate line. A background reader thread feeds stdout lines into a `queue.Queue`; `shell_exec` reads until it sees the sentinel and parses the exit code from it. A new queue is created each time the shell is restarted so reader threads from prior processes cannot contaminate the new session.

**Network:** Allowed (curl, pip, git, etc.). No allowlist — single trusted operator deployment.

**Env vars:**

| Var | Default | Purpose |
|---|---|---|
| `AGENT_WORKSPACE` | `<project-root>/workspace` | Workspace directory path |
| `AGENT_SHELL_TIMEOUT` | `30` | Default per-command timeout (seconds) |

### 5.4 Workspace edit tool (`tools/workspace_patch.py`)

| Tool | Description |
|---|---|
| `workspace_search_replace(path, old_string, new_string, replace_all?)` | Primary code-edit tool: exact substring replace in a file under `WORKSPACE` |

Paths are under the same rooted directory as `shell_exec` (`AGENT_WORKSPACE`). Read current file contents with `shell_exec` before editing and copy `old_string` verbatim (indentation/newlines must match). The tool rejects ambiguous matches unless `replace_all=true`. Writes are atomic (temp file + `os.replace`). Runs via `asyncio.to_thread` like other blocking sync tools.

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
- Receives: the last user message, the tool name, the tool arguments, and the full tool output (context for disambiguation only — the model is instructed not to echo these in its reply)
- Uses: **Gemma 4 26B** via `summarizer.summarize_gemma` — the same fast MoE model the `fetch_url` tool uses, authenticated with `GEMINI_API_KEY_FREE_RESOLVED` (see §2). Shared via the top-level `summarizer.py` module so there is exactly one summarizer implementation. The primary provider chain stays reserved for the agent loop.
- Returns: takeaways-only text (facts, numbers, names, errors, conclusions) — no restatement of the user request or tool args. The stored message is wrapped as `[history summary of <tool>]` + a one-line notice that the response was summarized for context efficiency + `Takeaways:` + that body (pagination notes for `fetch_url` are re-appended after the body when present)
- Is context-limited: the summarizer does NOT receive earlier conversation history, so summaries may be thin or generic if the goal was established several turns earlier (this is expected and noted in the system prompt)

**Non-blocking timing — DESIGN COMMITMENT.** Summarization MUST NEVER stall a turn. Concretely:

1. When a tool returns >8 000 chars, the agent starts the summary via `asyncio.create_task()` and keeps the **raw** content in the tool message for **all further LLM calls in that user turn**.
2. When the turn ends (final assistant reply with no tool calls, or max-iterations exit), `_apply_finished_summaries` runs once: a non-blocking poll (`await asyncio.sleep(0)`) swaps in the summary for tasks *already done*. It is **not** run between tool iterations — so chained tools always see prior raw results. In-flight tasks stay pending.
3. `agent.run()` returns `(response, provider, turn_messages, pending_summaries)`. `pending_summaries` is a list of `(message_dict, asyncio.Task)` pairs.
4. `main.py` appends the turn to SQLite with whatever content is currently in the dicts (raw, if the summary is still running) and then spawns `_finalize_summaries` as a fire-and-forget background task. When the summaries finish, that task mutates the in-memory `turn_messages` (the same dict objects cached per session) and calls `db.update_turn_messages(row_id, ...)` to overwrite the stored row.

The upshot: **the HTTP response returns the moment the model's final answer is ready**, regardless of how slow the summarizer is. The user never waits on history compaction. Subsequent turns see the summarized form as soon as the finalizer has run — typically within a second or two of the response, far before the user's next message.

**§4.1 ordering guard.** `_apply_finished_summaries` runs only at **turn end** (not between tool rounds), and the summary task never mutates the dict directly. Every in-turn `_call` therefore already sent the raw tool content to the API before any swap. Pinned by `test_raw_tool_content_preserved_even_when_summary_wins_race` and `test_multi_iteration_tools_prior_results_stay_raw_until_turn_end`.

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
| `POST` | `/chat/stream/cancel` | Cancel an active stream for `session_id`. Returns `{cancelled: bool}`. |
| `POST` | `/upload` | PDF text extraction (see §13). Returns `{filename, type, content}`. |
| `GET` | `/sessions` | List all sessions with preview text, message count, and last timestamp. |
| `GET` | `/sessions/{id}/history` | Display-friendly history for the UI (`get_display_history()`). |
| `DELETE` | `/sessions/{id}` | Delete a session from cache and SQLite. |
| `GET` | `/memories` | List all Mem0 memories. |
| `DELETE` | `/memories/{id}` | Delete a memory by ID. |
| `GET` | `/health` | Returns `{"status": "ok"}`. |
| `GET` | `/*` | Static files from `static/` (serves the web UI). |

### 8.1 Telegram bot transport (optional)

When `TELEGRAM_BOT_TOKEN` is set, `main.py` starts **long-polling** `getUpdates` in the FastAPI lifespan (`telegram_transport.py`). The bot shares the same SQLite history and `main.complete_chat_turn()` (non-streaming) as `POST /chat`.

**Access control**: Set `TELEGRAM_ALLOWED_USER_IDS` to a comma-separated list of Telegram **user** IDs (integers). When set, updates from anyone else are ignored (no reply, no LLM call). Uses `message.from.id`, so in groups only allowlisted senders can trigger the bot. When unset, any user who finds the bot can use it—set the allowlist in production.

**Session IDs**

- Default active session for a private/group chat: `tg:<chat_id>`.
- Forum topics / message threads: `tg:<chat_id>:<message_thread_id>`.
- `/new` creates `tg:<chat_id>:s-<slug>` (or the same with `:<thread>` before `:s-`) where **slug** is five **lowercase letters** (a–z) so you can `/switch` with just that slug when it is unique. The stored `session_id` still includes numeric `chat_id` for SQLite.

**Per-chat routing**: An in-memory map `(chat_id, thread_key) → active session_id` chooses which `session_id` receives plain text messages. It is **process-local** (lost on restart); sessions on disk remain, and `/sessions` + `/switch` recover them.

**Commands** (text starting with `/`): `/new`, `/sessions`, `/switch` (by **name**: short slug or `default`, or full internal `session_id` if needed), `/nuke` (instructs the model to call `nuke_chat`), `/help`, `/start`. Bot copy shows only the **slug** (or the label **default** for the base session), not the full `tg:…` id.

**Transport**: Bot messages use `sendMessage` with **`parse_mode: HTML`**. Incoming user text is still plain. No attachment forwarding from Telegram in this version.

**Live output**: normal chat messages are now streamed to Telegram by creating one placeholder message and updating it with `editMessageText` as chunks arrive. On completion, the edited message becomes the final reply (or a short notice + follow-up full send if the final text exceeds Telegram edit size comfort).

**Formatting (model)**: `agent.run(..., output_channel="telegram")` appends a Telegram addendum: plain text style, no markdown emphasis (`**`), no HTML tags, no fenced code, no LaTeX. Server-side send/edit sanitization also strips leaked thought tags (`<thought>/<thinking>/<redacted_*>`) and removes literal `**` before escaping HTML.

### 8.2 SSE event types (`/chat/stream`)

| `type` | Fields | Description |
|---|---|---|
| `text_chunk` | `text: str` | Incremental visible text from the model |
| `thinking_chars` | `count: int` | Number of thinking chars buffered so far (drives indicator) |
| `tool_call` | `name: str`, `args: dict` | A tool is about to be called |
| `tool_result` | `name: str`, `result: str` | Tool execution completed |
| `cancelled` | *(none)* | Stream was cancelled server-side (via `/chat/stream/cancel`) |
| `done` | `provider: str`, `turn_messages: list` | Turn complete; `turn_messages` is the full history slice |
| `error` | `detail: str` | Unrecoverable error |

After a successful turn, `main.py` extends the per-session in-memory cache and appends the row to SQLite (`_persist_stream_turn` runs in the stream producer’s `finally`, so this still happens if the browser disconnects mid-stream). The UI uses `turn_messages` from the `done` event to render the completed turn; canonical persisted history is served by `GET /sessions/{id}/history`.

**Server-owned streaming (per session):**

- **Multiple chats**: Any number of sessions may exist in parallel; each `session_id` has its own history and at most **one** active streamed turn at a time. A second `POST /chat/stream` for the same session while a turn is still running **attaches** to that turn’s event queue (duplicate generation is not started).
- **Client disconnect**: Closing the tab or losing the SSE connection **cancels only the HTTP response handler** for that browser; the background producer keeps running until the turn finishes, errors, or is cancelled via `/chat/stream/cancel`.
- **Persistence**: A completed, non-cancelled turn with a non-empty assistant reply is written to SQLite even when no client was connected at completion time (disconnect-safe persistence of the **final** turn — not mid-stream partials in the DB).

Cancellation semantics:

- `main.py` tracks one active streaming producer task per `session_id`.
- `POST /chat/stream/cancel` calls `task.cancel()` for that session and returns `{cancelled: true}` when a live stream existed.
- On cancellation, the SSE stream emits `{"type":"cancelled"}` and exits.
- Cancelled streams are **not persisted** to SQLite/history (same as error mid-stream).

---

## 9. Web UI

Single-page app (`static/index.html`). All state is managed client-side except conversation history (server-side in `_cache` / SQLite).

### 9.1 Layout

- **Left sidebar** (sessions panel): session list with preview text and timestamp; switch, create, delete sessions
- **Center** (chat): message feed + input bar (see §13 for file attach)
- **Right sidebar** (memories panel): list of all Mem0 memories with per-entry delete

### 9.2 Message rendering

**User messages**: plain text, pre-wrap, dark bubble.

**Agent messages**: markdown rendered via Marked.js with `breaks: true, gfm: true`. Before `marked.parse`, `escapeBracketMathDelimitersForMarked()` in `static/index.html` (1) doubles backslashes on `\(` `\)` `\[` `\]` outside fenced code so CommonMark does not strip them to plain brackets, and (2) inside those math segments normalizes broken row breaks: a lone `\` before optional spaces and a newline (Markdown hard break) becomes `\\`, and `\` + spaces before a digit, `(`, or `&` (common mistaken matrix row separator) becomes `\\`. During streaming, KaTeX runs on each repaint so previously-closed math stays rendered while later text is still arriving. LaTeX is rendered via KaTeX `auto-render` with bracket delimiters as the primary style (`\(...\)` inline, `\[...\]` display), with `$...$` tolerated as a compatibility fallback for occasional model slip-ups. Each text segment between tool calls gets its own bubble. Fenced code blocks are syntax-highlighted via **highlight.js** (GitHub theme) and include a per-block **Copy** button (hover-to-reveal on desktop; always visible on mobile) that writes the raw code text to the clipboard.

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

### 9.6 Keyboard shortcuts

Global shortcuts are implemented in `static/index.html` and intentionally work even when focus is in the contenteditable input:

- `Alt/Option + N` — new session (`newSession()`), then focus input.
- `Alt/Option + J` — toggle sessions panel.
- `Alt/Option + K` — toggle memories panel.
- `Alt/Option + L` — open file picker (`#file-input.click()`).
- `Alt/Option + 1..9` — switch to sessions index 1..9 from latest `/sessions` list.
- `Alt/Option + 0` — switch to sessions index 10.

Guardrails:
- Browser-native `Cmd/Ctrl` shortcuts are untouched (`Cmd+N`, `Cmd+1`, etc.).
- Shortcuts are ignored during IME composition and when `Ctrl` or `Cmd` is pressed.
- Option shortcuts are bound in capture phase and call `preventDefault()` + `stopPropagation()`. For macOS dead-key layouts, `keydown` also maps `Alt+Dead` + physical key (`N/J/K/L`) and `beforeinput`/`compositionend` scrub any queued dead-key residue in the contenteditable.
- Session switches (`switchSession`) await history load and then focus the input.

---

## 10. System Prompt

The system prompt is stored in `data/system_prompt.md` and read on every LLM call via `_build_system_prompt()` (so edits take effect immediately). The build step assembles three pieces:

1. `Today's date: YYYY-MM-DD` — prepended at request time so the model has a concrete present to reason against its January 2025 training cutoff. This is the structural anchor that makes the Trust & calibration section actually bind: without a known "today", post-cutoff stays abstract and the model's RLHF-trained reflex to disclaim recent info as possible hallucination tends to fire even on tool-grounded data.
2. The contents of `data/system_prompt.md` — the editable behavioral spec.
3. The tool documentation section generated from `TOOL_SCHEMAS`.

### 10.1 Current behavioral directives

The prompt is organized into nine sections; each one is short and independent so the file stays scannable and hackable. Section headings (in order):

- **Identity** — Zafir's personal AI assistant; optimize for correct + useful per minute of attention; direct, honest engagement over performed politeness; no hedging, no over-explaining, no trailing self-narration.
- **Today** — pointer to the dynamically injected date line; restates the January 2025 cutoff and that anything later comes from tools or this conversation, never from weights.
- **Trust & calibration** — the section that fixes the disclaim-as-hallucination bug. Names three distinct concepts that the model otherwise conflates (hallucination vs post-cutoff vs source error), explicitly acknowledges the RLHF training pull and tells the model to override it when it has tool data, frames tool results as "primary, not infallible" so source skepticism is preserved, adds the **`fetch_url` geometry rule** (128k summarizer window vs 8k-per-call raw chunks — missing facts in chunk 0 imply depth/pagination, not summarizer fabrication), forbids retroactively blanket-disclaiming past tool-grounded answers, and includes one worked before/after example.
- **Quality bar** — the general behavioral standard, modeled on what top-lab system prompts emphasize. One line each: calibration over hedging, no filler openers, honesty over compliance, no fabrication, verify before committing, finish what you start, reasoning depth proportional to task, format proportional to content, self-consistency within a turn.
- **Response style** — rendering specifics only: visible answer outside `<thought>/<thinking>/<redacted_*>` blocks, Markdown + KaTeX (`\(...\)` inline, `\[...\]` display; **no** `$` / `$$`), native `tool_calls` only.
- **System context** — one paragraph telling the model the runtime it operates in (multi-turn loop, streaming UI, tool-step rows, full history replay, repair call on empty visible output).
- **Context, turns, and tool results** — unified mental model: what a *turn* is; all tool results stay **raw for every LLM call in that turn**; compaction to `[history summary of <tool_name>]` after the turn ends (or via `main.py` finalizer); prior turns in replay show summaries; re-call the tool for verbatim raw on a new turn; `Takeaways:` body and summary accuracy / sparsity / `fetch_url` pagination hints.
- **Tool strategy** — concrete per-tool decision rules: `fetch_url` (prompt vs raw with explicit 128k vs 8k semantics, pagination, invalid verification: impeaching prompt output from raw chunk 0 alone), `recall` (silent on miss), `remember` (categories and what to store / not store), `list_memories` / `delete_memory`, `nuke_chat` (history reset with assistant summary when context bloat hurts performance), `shell_exec` with surgical reads (`rg` locate, `wc -l` size check, `sed -n`/`rg -n` slices over full dumps), then `workspace_search_replace` as the primary code-edit path (verbatim `old_string`). Keep edits narrow: choose short, unique snippets around target lines; avoid whole-file old/new replacements unless explicitly requested. Prefer thoughtful sequential calls over risky bulk edits; after a failed call, change arguments/approach instead of repeating the same failing call. **Tool payloads:** required arguments must be filled in the structured tool call—prose intent does not substitute; avoid `{}` and retry loops on missing-argument errors (mirror payload, verify fields, then retry once with a real fix).
- **Failure handling** — tool errors are data; retry, switch strategy, or report concisely; don't loop on a failing approach.

---

## 11. Known Limitations

This section lists remaining accepted trade-offs for this single-user deployment. Previously-flagged behavioral bugs (original §11.1 streaming repair and the summarizer model choice) have been fixed and are covered by regular green tests.

### 11.1 Streaming repair (resolved)

The streaming repair gate now keys on visible content (`visible_parts`), not raw content, so a stream consisting entirely of `<thinking>...</thinking>` correctly triggers the repair pass. The repair call runs against a scratch message list so the scaffold (`_REPAIR_USER` + the empty-visible assistant placeholder) does NOT leak into stored `turn_messages`. Verified by `tests/test_agent_loop.py::test_streaming_repair_triggers_on_thinking_only`.

### 11.2 Summarizer uses Gemma 26B (resolved)

`_summarize_for_history()` now uses Gemma 4 26B via the shared `summarizer.summarize_gemma` helper (~10× faster than the dense primary, fine for compact structured summaries). See §6.5. Verified by `tests/test_agent_helpers.py::test_summarize_for_history_uses_gemma_26b`.

---

## 13. File Uploads

Users can attach files to any message via drag-and-drop onto the chat area or the ⊕ button.

### 13.1 Supported types

| Type | Handling | Server round-trip? |
|---|---|---|
| Images (`image/*`) | `FileReader.readAsDataURL` in browser | No — base64 data URL held in JS state |
| Text / code files (`text/*` or known extensions) | `FileReader.readAsText` in browser | No — UTF-8 text held in JS state |
| PDFs (`application/pdf`) | `POST /upload` → `pypdf` extraction → text returned | Yes — pypdf runs server-side |

Size cap: **5 MB per file**. Rejected with HTTP 413 at the `/upload` endpoint; enforced client-side for browser-handled types.

### 13.2 API contract

`POST /chat` and `POST /chat/stream` accept an optional `attachments` list:

```json
{
  "message": "explain this",
  "session_id": "...",
  "attachments": [
    {"type": "text",  "filename": "foo.py",    "content": "def f(): pass"},
    {"type": "image", "filename": "shot.png",  "content": "data:image/png;base64,..."}
  ]
}
```

`main._build_user_content()` converts this to an OpenAI vision-format content list:
- Text attachments → `{"type": "text", "text": "[File: <name>]\n<content>"}` (prepended before the message)
- Image attachments → `{"type": "image_url", "image_url": {"url": "<data-url>"}}`
- User's text message → `{"type": "text", "text": "<message>"}` (appended last)

When there are no attachments, `user_content` is a plain string — the API contract is unchanged for existing callers.

### 13.3 Agent integration

`agent.run()` and `agent.run_stream()` accept `user_content: str | list` instead of the former `user_message: str`. The internal `_user_content_as_text()` helper extracts a plain-string representation for use in history summarization prompts. `_sanitize_message()` already passes list content through for the user role.

### 13.4 History storage

The raw multimodal content list is stored in `turn_messages` (JSON-serialized in SQLite) and replayed to the LLM on subsequent turns. `db.get_display_history()` flattens list content to a plain string for the UI — image parts become `[image]`, text parts are joined.

The `content` column (used for session preview) always stores the text portion of the message (`req.message`), not the full multimodal content.

### 13.5 UI

The input is a `contenteditable` div instead of a `<textarea>`. File chips (`contenteditable="false"` inline `<span>` elements) sit inline with text — the browser treats them as characters, so cursor navigation, Backspace, and Delete remove them natively. Chips show a type symbol (`📄`/`🖼`/`📑`) and a truncated filename. Sent user messages show read-only chip indicators in the bubble.

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
| Summarizer finishes *before* the next in-turn LLM call | Summary is NOT swapped in until **turn end** — every in-turn `_call` sees full prior tool output; mid-turn swap would regress §4.1 | `test_raw_tool_content_preserved_even_when_summary_wins_race`, `test_multi_iteration_tools_prior_results_stay_raw_until_turn_end` |
