"""System-prompt read/write tool tests. Uses tmp_system_prompt so the real
`data/system_prompt.md` is never mutated.
"""
from __future__ import annotations

from tools import self_modify


def test_read(tmp_system_prompt):
    content = self_modify.get_system_prompt()
    assert "Test system prompt" in content


def test_edit_roundtrip(tmp_system_prompt):
    result = self_modify.edit_system_prompt("New prompt body", "because test")
    assert "updated" in result.lower()
    assert "because test" in result
    assert self_modify.get_system_prompt() == "New prompt body"


def test_edit_is_destructive(tmp_system_prompt):
    """edit_system_prompt overwrites, not appends (DESIGN §5.3)."""
    self_modify.edit_system_prompt("first", "r1")
    self_modify.edit_system_prompt("second", "r2")
    assert self_modify.get_system_prompt() == "second"
