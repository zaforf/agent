"""Contextual query service for ephemeral, user-facing explanations.

Used by:
- Highlight-to-Explain: Shift+highlight passes selection + surrounding chat
  context; implicit task is "explain this."
- /btw (future): user asks a side question; explicit query overrides the task.

Interface: contextual_query(selection, context, query="") → str
"""
from __future__ import annotations

from openai import AsyncOpenAI

from config import GEMINI_API_KEY, GEMINI_BASE_URL

_MODEL = "gemini-3.1-flash-lite-preview"

_SYSTEM = (
    "You are a concise inline assistant. The user has highlighted a piece of text "
    "from a conversation. Explain or answer about the highlighted selection using the "
    "surrounding context.\n"
    "Rules:\n"
    "- 2-3 sentences maximum.\n"
    "- Start directly with the substance — no 'Here is...', 'This refers to...', "
    "'Sure!', or any filler opener.\n"
    "- If it's a term or concept, define it. If it's code, say what it does. "
    "If it's an acronym, expand it.\n"
    "- Plain prose only. No bullet lists, no headers, no markdown."
)


async def contextual_query(
    selection: str,
    context: str,
    query: str = "",
) -> str:
    """Call the flash model with selection + surrounding context.

    `query` is optional — omit it (highlight case) for the implicit
    "explain the selection" task. Pass it for /btw-style queries.
    """
    client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    task = query.strip() or "Explain the highlighted text."
    user_prompt = f"Context:\n{context}\n\nHighlighted: {selection!r}\n\nTask: {task}"
    resp = await client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=256,
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()
