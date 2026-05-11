# Agent

High-leverage AI assistant for technical work, research, and automation. Built as a Telegram bot with specialized tools for web retrieval, long-context synthesis, and workspace interaction.

## Core Capabilities

- **Deep Web Research**: Uses Brave Search and a long-context summarizer (Gemma 26B) to extract targeted information from URLs without context flooding.
- **Workspace Automation**: Direct shell access, file I/O, and precise search-and-replace for code editing.
- **Long-term Memory**: Vector-based durable memory to track user preferences, project state, and learning progress across sessions.
- **YouTube Integration**: Transcript extraction and summarization via Supadata.

## Architecture

- **Transport**: Telegram Bot API.
- **Brain**: Gemma 4 / Gemini family (via API).
- **Memory**: Durable vector store for cross-session continuity.
- **Tools**: 
    - `web_search`: Discover URLs.
    - `fetch_url`: Precise extraction from pages.
    - `youtube_transcript`: Targeted analysis of video content.
    - `shell_exec`: General purpose bash automation.
    - `workspace_*`: Targeted file reads and edits.

## Setup

1. `pip install -r requirements.txt`
2. Configure `.env` with:
    - `TELEGRAM_BOT_TOKEN`
    - `BRAVE_SEARCH_API_KEY`
    - `SUPADATA_API_KEY`
    - `GEMINI_API_KEY`
3. Run `python main.py`

