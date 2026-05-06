"""
Tool registry. To add a new tool:
  1. Create a module in tools/ with SCHEMAS and FUNCTIONS dicts
  2. Import and register it here — that's it.
"""

from tools import memory, nuke, shell, web, workspace_patch, youtube

TOOL_SCHEMAS: list[dict] = [
    *memory.SCHEMAS,
    *nuke.SCHEMAS,
    *web.SCHEMAS,
    *shell.SCHEMAS,
    *workspace_patch.SCHEMAS,
    *youtube.SCHEMAS,
]

TOOL_FUNCTIONS: dict[str, callable] = {
    **memory.FUNCTIONS,
    **nuke.FUNCTIONS,
    **web.FUNCTIONS,
    **shell.FUNCTIONS,
    **workspace_patch.FUNCTIONS,
    **youtube.FUNCTIONS,
}
