"""Tool for resetting long chat history to a compact assistant summary."""
from __future__ import annotations

import json

_NUKE_PREFIX = "__NUKE_CHAT__"


def nuke_chat(summary: str) -> str:
    """Request a history reset with a retained assistant summary.

    The actual reset is applied by API-layer handlers (which know session/cache/DB).
    This tool returns a structured marker that agent.py can detect.
    """
    text = (summary or "").strip()
    if not text:
        raise ValueError("summary must be non-empty")
    payload = json.dumps({"summary": text}, ensure_ascii=False)
    return _NUKE_PREFIX + payload


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "nuke_chat",
            "description": (
                "Reset the current chat history to a single assistant summary message. "
                "Use only when history bloat is hurting performance. Pass a concise but sufficient "
                "summary of everything important for continuing the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Sufficient continuation summary for the next turns",
                    }
                },
                "required": ["summary"],
            },
        },
    }
]

FUNCTIONS = {"nuke_chat": nuke_chat}
