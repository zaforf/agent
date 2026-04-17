"""Real-network tests for tools/fetch.py.

Marked `@pytest.mark.net` so the default `pytest` run skips them. Run with
`pytest -m net` when you want to exercise the real httpx call path (User-Agent,
retries, TLS, HTML extraction).

Kept minimal and targeted at rock-stable endpoints — example.com is defined by
RFC 2606 and is intentionally static.
"""
from __future__ import annotations

import pytest

from tools import fetch


pytestmark = pytest.mark.net


def test_fetch_example_com_raw_extracts_body_text():
    """Real fetch against example.com in raw mode — no LLM, just the HTTP +
    extractor path. Verifies:
      - TLS + User-Agent handshake succeeds
      - HTML parses and script/style tags are stripped
      - The RFC-stable body text comes through
    """
    out = fetch.fetch_url("https://example.com", raw=True)
    assert "Example Domain" in out, f"body text missing: {out[:300]!r}"
    # The extractor should drop HTML plumbing
    assert "<html" not in out.lower()
    assert "<script" not in out.lower()


def test_fetch_example_com_short_page_has_no_pagination_note():
    """example.com is ~1kB, well below _RAW_CHAR_LIMIT — no pagination note."""
    out = fetch.fetch_url("https://example.com", raw=True)
    assert "call fetch_url with offset=" not in out
