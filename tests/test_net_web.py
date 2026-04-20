"""Real-network tests for tools/web.py.

Marked `@pytest.mark.net` so the default `pytest` run skips them. Run with
`pytest -m net` when you want to exercise the real httpx call path (User-Agent,
retries, TLS, HTML extraction) and the live Brave Search API.

Kept minimal and targeted at rock-stable endpoints — example.com is defined by
RFC 2606 and is intentionally static. The web_search test is skipped when
BRAVE_SEARCH_API_KEY isn't set.
"""
from __future__ import annotations

import pytest

import config
from tools import web as fetch


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


@pytest.mark.skipif(not config.BRAVE_SEARCH_API_KEY, reason="BRAVE_SEARCH_API_KEY not set")
def test_web_search_brave_returns_results_for_python_org():
    """Live Brave Search — query should return at least one result and the
    numbered markdown format the tool advertises.
    """
    out = fetch.web_search("python.org official site", max_results=3)
    assert not out.startswith("Error:"), out
    assert "1. **" in out, f"missing numbered markdown: {out[:300]!r}"
    assert "https://" in out, f"no http URL in response: {out[:300]!r}"
