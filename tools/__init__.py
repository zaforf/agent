"""
Tool registry. To add a new tool:
  1. Create a module in tools/ with SCHEMAS and FUNCTIONS dicts
  2. Import and register it here — that's it.
"""

from tools import continuation, lsp_navigation, memory, nuke, shell, web, workspace_patch, youtube

TOOL_SCHEMAS: list[dict] = [
    *continuation.SCHEMAS,
    *memory.SCHEMAS,
    *nuke.SCHEMAS,
    *web.SCHEMAS,
    *shell.SCHEMAS,
    *workspace_patch.SCHEMAS,
    *lsp_navigation.SCHEMAS,
    *youtube.SCHEMAS,
]

TOOL_FUNCTIONS: dict[str, callable] = {
    **continuation.FUNCTIONS,
    **memory.FUNCTIONS,
    **nuke.FUNCTIONS,
    **web.FUNCTIONS,
    **shell.FUNCTIONS,
    **workspace_patch.FUNCTIONS,
    **lsp_navigation.FUNCTIONS,
    **youtube.FUNCTIONS,
}
