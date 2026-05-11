# Agent

Telegram bot with tools for web research, long-context synthesis, and workspace interaction.

## Tools

* `web_search`: Brave Search for discovering URLs.
* `fetch_url`: Precise extraction from pages using Gemma 26B summarizer.
* `youtube_transcript`: Targeted analysis of video content via Supadata.
* `shell_exec`: General purpose bash automation.
* `workspace_*`: Targeted file reads and edits.

## Setup

1. `pip install -r requirements.txt`
2. Configure `.env`:
    - `TELEGRAM_BOT_TOKEN`
    - `BRAVE_SEARCH_API_KEY`
    - `SUPADATA_API_KEY`
    - `GEMINI_API_KEY`
3. Run `python main.py`
