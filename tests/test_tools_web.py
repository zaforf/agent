"""Unit tests for tools/web.py — HTML extraction, pagination math, summarizer
fallback, and web_search (Brave). HTTP is fully mocked via monkeypatch;
no network in default pytest.
"""
from __future__ import annotations

import httpx

from tools import web as fetch  # historical alias keeps fetch_url tests terse


# ── _TextExtractor ───────────────────────────────────────────────────────────

def test_text_extractor_strips_script_and_style():
    html = """
    <html><body>
      <script>var x = 1;</script>
      <style>body { color: red; }</style>
      <p>Visible paragraph.</p>
    </body></html>
    """
    ex = fetch._TextExtractor()
    ex.feed(html)
    text = ex.get_text()
    assert "var x" not in text
    assert "color: red" not in text
    assert "Visible paragraph." in text


def test_text_extractor_strips_nav_header_footer():
    html = """
    <nav>nav links</nav>
    <header>header bar</header>
    <footer>footer notice</footer>
    <p>Main content.</p>
    """
    ex = fetch._TextExtractor()
    ex.feed(html)
    text = ex.get_text()
    assert "nav links" not in text
    assert "header bar" not in text
    assert "footer notice" not in text
    assert "Main content." in text


def test_text_extractor_inserts_block_newlines():
    html = "<p>A</p><p>B</p>"
    ex = fetch._TextExtractor()
    ex.feed(html)
    text = ex.get_text()
    # There should be at least one newline between the two paragraphs.
    assert "A" in text and "B" in text
    assert "\n" in text


# ── Pagination math (raw mode) ───────────────────────────────────────────────

class _FakeResp:
    def __init__(self, text, content_type="text/html"):
        self.text = text
        self.headers = {"content-type": content_type}


def _mock_fetch(monkeypatch, text, content_type="text/html"):
    def _fake(url, headers):
        return _FakeResp(text, content_type)
    monkeypatch.setattr(fetch, "_fetch_with_retries", _fake)


def test_raw_mode_returns_chunk_with_pagination_note(monkeypatch):
    # Build 3× the raw limit of plain text (no HTML parsing side-effects)
    big = "A" * (fetch._RAW_CHAR_LIMIT * 3)
    _mock_fetch(monkeypatch, big, content_type="text/plain")
    out = fetch.fetch_url("https://example.com", raw=True)
    assert out.startswith("A" * 100)  # starts with raw text
    assert "call fetch_url with offset=" in out, f"no pagination note: {out[-200:]!r}"
    assert f"offset={fetch._RAW_CHAR_LIMIT}" in out


def test_raw_mode_offset_arithmetic(monkeypatch):
    big = "B" * (fetch._RAW_CHAR_LIMIT * 3)
    _mock_fetch(monkeypatch, big, content_type="text/plain")
    # Start mid-way through
    out = fetch.fetch_url("https://example.com", raw=True, offset=fetch._RAW_CHAR_LIMIT)
    assert f"offset={fetch._RAW_CHAR_LIMIT * 2}" in out, out[-200:]


def test_raw_mode_no_note_when_below_limit(monkeypatch):
    short = "hello world"
    _mock_fetch(monkeypatch, short, content_type="text/plain")
    out = fetch.fetch_url("https://example.com", raw=True)
    assert out == short
    assert "call fetch_url" not in out


def test_summarizer_failure_falls_back_to_raw(monkeypatch):
    """When the summarizer blows up, fetch_url labels the raw fallback."""
    text = "hello " * 100
    _mock_fetch(monkeypatch, text, content_type="text/plain")

    def _boom(text, prompt):
        raise RuntimeError("summarizer unavailable")
    monkeypatch.setattr(fetch, "_summarize_content", _boom)

    out = fetch.fetch_url("https://example.com", prompt="extract something")
    assert "hello" in out, "expected raw text fallback on summarizer failure"
    assert "structured page summarization was unavailable" in out


def test_empty_page_returns_no_content_marker(monkeypatch):
    _mock_fetch(monkeypatch, "   \n  ", content_type="text/plain")
    out = fetch.fetch_url("https://example.com", raw=True)
    assert out == "(no readable content)"


def test_schema_raw_param_uses_constant():
    """The tool schema should advertise the same raw-char limit as the code."""
    (schema,) = [s for s in fetch.SCHEMAS if s["function"]["name"] == "fetch_url"]
    raw_desc = schema["function"]["parameters"]["properties"]["raw"]["description"]
    assert f"{fetch._RAW_CHAR_LIMIT:,}" in raw_desc


# ── web_search (Brave) ───────────────────────────────────────────────────────

class _FakeSearchResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("GET", fetch._BRAVE_URL)
            raise httpx.HTTPStatusError(
                "boom", request=req,
                response=httpx.Response(self.status_code, request=req),
            )

    def json(self):
        return self._payload


class _FakeSearchClient:
    def __init__(self, captured: dict, payload: dict, status: int = 200):
        self._captured = captured
        self._payload = payload
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def get(self, url, headers=None, params=None):
        self._captured["url"] = url
        self._captured["headers"] = dict(headers or {})
        self._captured["params"] = dict(params or {})
        return _FakeSearchResp(self._payload, status=self._status)


def _patch_search(monkeypatch, payload, status=200, key="k-brave"):
    captured: dict = {}
    monkeypatch.setattr(fetch, "BRAVE_SEARCH_API_KEY", key)
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **k: _FakeSearchClient(captured, payload, status),
    )
    return captured


def test_web_search_returns_numbered_markdown_results(monkeypatch):
    _patch_search(monkeypatch, {"web": {"results": [
        {"title": "Python", "url": "https://python.org",
         "description": "<strong>Python</strong> language"},
        {"title": "PyPI", "url": "https://pypi.org",
         "description": "Package index"},
    ]}})
    out = fetch.web_search("python", max_results=2)
    assert out.startswith("1. Python - https://python.org"), out
    assert "2. PyPI - https://pypi.org" in out
    assert "<strong>" not in out, "html highlight tags must be stripped"




def test_web_search_strips_control_characters(monkeypatch):
    _patch_search(monkeypatch, {"web": {"results": [
        {"title": "A\u0000B", "url": "https://e\u0007xample.com", "description": "line\u000bbreak"},
    ]}})
    out = fetch.web_search("q")
    assert "\x00" not in out and "\x07" not in out and "\x0b" not in out


def test_web_search_total_output_truncated(monkeypatch):
    monkeypatch.setattr(fetch, "_SEARCH_TOTAL_CHARS", 80)
    _patch_search(monkeypatch, {"web": {"results": [
        {"title": "Long title", "url": "https://example.com", "description": "x" * 500},
    ]}})
    out = fetch.web_search("q")
    assert out.endswith("… [truncated]")

def test_web_search_clamps_max_results(monkeypatch):
    captured = _patch_search(monkeypatch, {"web": {"results": []}})
    fetch.web_search("q", max_results=fetch._SEARCH_MAX_RESULTS * 100)
    assert captured["params"]["count"] == fetch._SEARCH_MAX_RESULTS


def test_web_search_no_key_returns_stable_error(monkeypatch):
    monkeypatch.setattr(fetch, "BRAVE_SEARCH_API_KEY", "")
    out = fetch.web_search("anything")
    assert out.startswith("Error: web_search disabled")


def test_web_search_empty_query_returns_error(monkeypatch):
    monkeypatch.setattr(fetch, "BRAVE_SEARCH_API_KEY", "k")
    assert fetch.web_search("   ").startswith("Error: web_search needs")


def test_web_search_no_results_message(monkeypatch):
    _patch_search(monkeypatch, {"web": {"results": []}})
    assert "No results" in fetch.web_search("zzz")


def test_web_search_http_error_returns_string(monkeypatch):
    _patch_search(monkeypatch, {}, status=429)
    out = fetch.web_search("q")
    assert out.startswith("Error: web_search HTTP 429"), out


def test_web_search_quota_error_is_actionable(monkeypatch):
    _patch_search(monkeypatch, {}, status=402)
    out = fetch.web_search("q")
    assert "quota exhausted" in out
    assert "new Brave Search plan or API key" in out


def test_web_search_sends_subscription_token(monkeypatch):
    captured = _patch_search(monkeypatch, {"web": {"results": []}}, key="brave-secret")
    fetch.web_search("q")
    assert captured["headers"].get("X-Subscription-Token") == "brave-secret"
