"""Tool registry consistency tests.

Adding a new tool is defined in DESIGN §5 as creating a module with SCHEMAS +
FUNCTIONS and importing it in tools/__init__.py. These tests guard that
invariant so a mismatch is caught before a live call fails mysteriously.
"""
from __future__ import annotations

import pytest

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


def test_every_function_has_a_schema():
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    fn_names = set(TOOL_FUNCTIONS)
    missing = fn_names - schema_names
    assert not missing, f"functions with no schema: {missing}"


def test_every_schema_has_a_function():
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    fn_names = set(TOOL_FUNCTIONS)
    orphan = schema_names - fn_names
    assert not orphan, f"schemas with no callable: {orphan}"


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["function"]["name"])
def test_schema_shape(schema):
    assert schema.get("type") == "function"
    fn = schema["function"]
    assert fn.get("name")
    assert fn.get("description"), f"{fn['name']}: missing description"
    params = fn.get("parameters")
    assert isinstance(params, dict) and params.get("type") == "object"
    assert "properties" in params
    # `required` is optional per JSON Schema but if present must be a list
    if "required" in params:
        assert isinstance(params["required"], list)


def test_summarize_text_tool_not_registered():
    """The legacy opt-in summarizer tool was removed; summarization is automatic."""
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert "summarize_text" not in names
    assert "summarize_text" not in TOOL_FUNCTIONS


def test_core_tools_are_present():
    """These are the tools DESIGN §5 names by hand."""
    required = {
        "remember", "recall", "list_memories", "delete_memory",
        "fetch_url", "web_search",
        "get_system_prompt", "edit_system_prompt",
        "shell_exec",
    }
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    missing = required - names
    assert not missing, f"missing core tools: {missing}"
