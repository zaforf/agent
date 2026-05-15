"""Continuation tool — keeps the agent loop alive between sub-tasks."""
from __future__ import annotations


def continue_task(next_steps: str) -> str:
    """Signal that there is more work to do and stay in the loop.

    Call this instead of outputting a text summary of planned next steps.
    The loop continues and you should immediately begin executing next_steps.
    """
    return f"(continuing) {next_steps}"


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "continue_task",
            "description": (
                "Stay in the loop when you have more work to do. "
                "Call this instead of ending your turn with a text description of next steps. "
                "Pass your immediate plan as next_steps — the loop will continue and you "
                "must begin executing it in the very next tool call. "
                "Do NOT call this when the task is genuinely complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "next_steps": {
                        "type": "string",
                        "description": "What you will do next — be specific and actionable",
                    },
                },
                "required": ["next_steps"],
            },
        },
    }
]

FUNCTIONS = {"continue_task": continue_task}
