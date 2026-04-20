"""
Tool registry. To add a new tool:
  1. Create a module in tools/ with SCHEMAS and FUNCTIONS dicts
  2. Import and register it here — that's it.
"""

from tools import memory, self_modify, web

TOOL_SCHEMAS: list[dict] = [
    *memory.SCHEMAS,
    *self_modify.SCHEMAS,
    *web.SCHEMAS,
]

TOOL_FUNCTIONS: dict[str, callable] = {
    **memory.FUNCTIONS,
    **self_modify.FUNCTIONS,
    **web.FUNCTIONS,
}
