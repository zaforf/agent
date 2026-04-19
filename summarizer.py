"""Shared helper for posting prompts to the Gemma 4 26B summarizer endpoint.

Used by:
- `tools/fetch.py`'s `_summarize_content` (extract-from-page summarization)
- `agent._summarize_for_history` (compact tool-result summaries for history)

Both paths want the same thing: a fast, cheap, long-context model that isn't
the primary agent. `gemma-4-26b-a4b-it` is an MoE with 3.8B active params —
latency in the ~1-2s range at 8k output tokens, ~10x faster than the primary.

History summarization (`agent._summarize_for_history`) passes its own
`systemInstruction` — takeaways-only, no echo of user/tool metadata — while
`fetch_url` uses an extraction-oriented instruction.

This module is dependency-light (only `httpx` + stdlib) and does a blocking
HTTP call. Call from async contexts via `asyncio.to_thread(summarize_gemma, ...)`.
"""
from __future__ import annotations

import time

import httpx

import config

MODEL = "gemma-4-26b-a4b-it"
_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

_HTTP_TIMEOUT_S = 60
_RETRIES = 3


def summarize_gemma(
    system_instruction: str,
    user_prompt: str,
    max_output_tokens: int = 8192,
) -> str:
    """Blocking call to Gemma 26B. Returns the concatenated non-thought text
    from the first candidate. Retries on 429 with exponential backoff.

    Raises `RuntimeError` if every retry fails or the API returns no candidates.
    """
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }

    key = config.GEMINI_API_KEY_FREE_RESOLVED
    with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
        for attempt in range(_RETRIES):
            resp = client.post(_URL, params={"key": key}, json=payload)
            if resp.status_code == 429 and attempt < _RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            candidates = resp.json().get("candidates", [])
            if not candidates:
                raise RuntimeError("summarizer: no candidates in response")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(
                p["text"] for p in parts
                if "text" in p and not p.get("thought")
            )
            return text.strip()

    raise RuntimeError("summarizer: all retries exhausted")
