"""
Tool registry. To add a new tool:
  1. Create a module in tools/ with SCHEMAS and FUNCTIONS dicts
  2. Import and register it here — that's it.
"""

from tools import memory, shell, web, workspace_patch

TOOL_SCHEMAS: list[dict] = [
    *memory.SCHEMAS,
    *web.SCHEMAS,
    *shell.SCHEMAS,
    *workspace_patch.SCHEMAS,
]

TOOL_FUNCTIONS: dict[str, callable] = {
    **memory.FUNCTIONS,
    **web.FUNCTIONS,
    **shell.FUNCTIONS,
    **workspace_patch.FUNCTIONS,
}
