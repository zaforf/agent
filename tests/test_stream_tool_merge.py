"""Unit tests for streaming tool-call fragment merging (agent._merge_stream_fragment).

OpenAI's Chat Completions streaming schema exposes ``choices[].delta.tool_calls[]``
with optional ``function.name`` and ``function.arguments`` per chunk. The docs
describe these fields as arriving with the streamed completion, but do **not**
guarantee that each chunk is a strict suffix-only delta for every provider.

In the wild, some OpenAI-compatible stacks re-send the **full** function name on
later chunks for the same tool-call index (see e.g. `open-webui/open-webui#22177`).
Naive ``current + fragment`` accumulation then produces ``web_searchweb_search``.

These tests pin the merge semantics we rely on so regressions are obvious.
"""
from __future__ import annotations

import agent


def test_merge_empty_and_single_fragment():
    assert agent._merge_stream_fragment("", "") == ""
    assert agent._merge_stream_fragment("", "x") == "x"
    assert agent._merge_stream_fragment("ab", "") == "ab"


def test_merge_true_suffix_delta():
    """Typical case: arguments stream as successive suffix fragments."""
    assert agent._merge_stream_fragment('{"q":', '"x"}') == '{"q":"x"}'


def test_merge_cumulative_snapshot_extends_prefix():
    """Later chunk repeats prior text then adds more (prefix extension)."""
    assert agent._merge_stream_fragment("web_", "web_search") == "web_search"


def test_merge_resend_full_value_is_idempotent():
    """Provider sends the same full name again while arguments continue."""
    assert agent._merge_stream_fragment("web_search", "web_search") == "web_search"


def test_merge_shorter_fragment_is_ignored_when_already_longer():
    """Stale/partial chunk should not truncate a longer accumulated value."""
    assert agent._merge_stream_fragment("web_search", "web_") == "web_search"


def test_merge_arguments_cumulative_snapshot():
    """Arguments sometimes widen in place (fragment is a strict extension of current)."""
    assert agent._merge_stream_fragment('{"query":"h', '{"query":"hi"}') == '{"query":"hi"}'


def test_merge_open_webui_reported_pattern():
    """Regression shape from community reports: full name in consecutive chunks."""
    a = "my_server_search"
    assert agent._merge_stream_fragment(a, a) == a
