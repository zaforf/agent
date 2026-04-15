from config import SYSTEM_PROMPT_PATH


def get_system_prompt() -> str:
    """Read the current system prompt."""
    return SYSTEM_PROMPT_PATH.read_text()


def edit_system_prompt(new_prompt: str, reason: str) -> str:
    """Overwrite the system prompt file with new content."""
    SYSTEM_PROMPT_PATH.write_text(new_prompt)
    return f"System prompt updated. Reason: {reason}"


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_prompt",
            "description": (
                "Read the current system prompt in full. "
                "You MUST call this before edit_system_prompt — never edit blind."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_system_prompt",
            "description": (
                "Overwrite the system prompt. Rules: "
                "(1) Always call get_system_prompt first. "
                "(2) Make surgical edits — preserve existing content, only change what's needed. "
                "(3) Never rewrite entirely unless the user explicitly asks. "
                "(4) Only use when user feedback clearly requires a permanent behavior change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_prompt": {
                        "type": "string",
                        "description": "Full replacement content for the system prompt",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this change is needed",
                    },
                },
                "required": ["new_prompt", "reason"],
            },
        },
    },
]

FUNCTIONS = {
    "get_system_prompt": get_system_prompt,
    "edit_system_prompt": edit_system_prompt,
}
