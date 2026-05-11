# Agent

Personal AI assistant system featuring a multi-turn agentic loop, long-term vector memory, and deep workspace integration.

## Architecture

- **Agent Loop**: Multi-turn execution (up to 30 iterations) with native tool use.
- **Provider Cascade**: Tiered fallback chain (Gemma 4 $\rightarrow$ Cerebras $\rightarrow$ Groq) with exponential backoff.
- **Context Management**: 
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
* `workspace_*`: Surgical search-and-replace file editing.

## Setup

1. `pip install -r requirements.txt`
2. Configure `.env`:
    - `TELEGRAM_BOT_TOKEN`, `BRAVE_SEARCH_API_KEY`, `SUPADATA_API_KEY`, `GEMINI_API_KEY`
3. Run `python main.py`
