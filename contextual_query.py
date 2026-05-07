"""Contextual query service for ephemeral, user-facing explanations.

Used by:
- Highlight-to-Explain: Shift+highlight passes selection + surrounding chat
  context; implicit task is "explain this."
- /btw (future): user asks a side question; explicit query overrides the task.

Interface: contextual_query(selection, context, query="") → str
"""
from __future__ import annotations

from openai import AsyncOpenAI

from config import GROQ_API_KEY

_MODEL = "llama-3.3-70b-versatile"
_BASE_URL = "https://api.groq.com/openai/v1"

_SYSTEM = (
    "You are an inline knowledge assistant. The user highlighted a word or phrase "
    "from a conversation and wants useful context they don't already have.\n\n"
    "Rules:\n"
    "- Never repeat or paraphrase what is already stated in the surrounding context — "
    "the user can read it. Add something new.\n"
    "- Prioritise: origin and history (when/why coined or invented), the broader field "
    "or movement it belongs to, who created it and why it matters, common "
    "misconceptions, or how it relates to adjacent concepts.\n"
    "- For acronyms: expand, then give origin/purpose — not just a definition.\n"
    "- For jargon or technical terms: explain the intuition and context of use, "
    "not just the dictionary meaning.\n"
    "- Be genuinely useful. A response that only restates the selection in different "
    "words is a failure.\n"
    "- 2-4 sentences. Dense with insight, not padded prose.\n"
    "- No filler openers ('This refers to...', 'Sure!', 'Here is...'). Start with "
    "the substance.\n"
    "- Plain prose only. No bullets, headers, or markdown."
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
    client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=_BASE_URL)
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
