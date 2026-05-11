# Agent

Personal AI assistant system featuring a multi-turn agentic loop, long-term vector memory, and deep workspace integration.

## Architecture

- **Agent Loop**: Multi-turn agent loop (up to 30 tool iterations per turn) with native tool use.
- **Provider Cascade**: Tiered fallback chain (Gemma 4 $\rightarrow$ Cerebras $\rightarrow$ Groq) with exponential backoff.
- **Context Management**: 
    - **Ambient Prefetch**: Automatic retrieval of relevant long-term memories before the first LLM call of every turn.
    - **In-turn**: Tool results are kept raw for all calls within a single turn.
    - **Post-turn**: Large tool outputs (>8k chars) are asynchronously summarized via Gemma 26B to optimize history context without blocking the response.
- **Transport**: 
    - **Web UI**: Vanilla JS + SSE for real-time streaming of text, thinking blocks, and tool steps.
    - **Telegram**: Long-polling bot transport with live-edit streaming.
- **Memory**: Durable vector store (Mem0 + Qdrant) for cross-session continuity and preference tracking.

## Tools

* `web_search`: Brave Search for URL discovery.
* `fetch_url`: Targeted extraction using a long-context summarizer.
* `youtube_transcript`: Transcript analysis via Supadata.
* `shell_exec`: Persistent bash session with CWD tracking.
* **Workspace Workflow**: `workspace_grep` (locate) $\rightarrow$ `workspace_read` (read) $\rightarrow$ `workspace_search_replace` (edit).

## UI Features

- **Highlight-to-Explain**: Shift+select text in the chat to trigger a targeted explanation turn.
- **Token Counter**: Live monitoring of context window usage.
- **Math Rendering**: KaTeX support for inline and display equations.

## Setup

1. `pip install -r requirements.txt`
2. Configure `.env`:
    - `TELEGRAM_BOT_TOKEN`, `BRAVE_SEARCH_API_KEY`, `SUPADATA_API_KEY`, `GEMINI_API_KEY`
3. Run `python main.py`
