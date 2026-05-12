"""Unit tests for tools/youtube.py — URL parsing, pagination, error handling.

HTTP is fully mocked; no network required for default pytest run.
"""
from __future__ import annotations

import pytest

import config
from tools import youtube as yt


# ── _to_url ──────────────────────────────────────────────────────────────────

def test_to_url_bare_id():
    assert yt._to_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_to_url_watch_url():
    assert yt._to_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") ==         "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_to_url_watch_url_extra_params():
    # Extra query params before v=
    assert yt._to_url("https://www.youtube.com/watch?list=PL&v=dQw4w9WgXcQ") ==         "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_to_url_short_url():
    assert yt._to_url("https://youtu.be/dQw4w9WgXcQ") ==         "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_to_url_shorts_url():
    assert yt._to_url("https://www.youtube.com/shorts/dQw4w9WgXcQ") ==         "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_to_url_embed_url():
    assert yt._to_url("https://www.youtube.com/embed/dQw4w9WgXcQ") ==         "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_to_url_unknown_passes_through():
    # Not a recognized pattern — pass through so the API can surface the error.
    s = "not-a-youtube-thing"
    assert yt._to_url(s) == s


# ── youtube_transcript (mocked Supadata) ─────────────────────────────────────

class _FakeTranscript:
    def __init__(self, content):
        self.content = content


class _FakeMetadata:
    def __init__(self):
        self.content = {
            "title": "Test Title",
            "author": {"display_name": "Test Channel"},
            "description": "Test Description",
        }


class _FakeSupadataClient:
    def __init__(self, content="hello world transcript"):
        self._content = content

    def metadata(self, url):
        return _FakeMetadata()

    def transcript(self, url, text=False):
        return _FakeTranscript(self._content)


def _patch(monkeypatch, content="hello world transcript"):
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "test-key")
    monkeypatch.setattr(yt, "Supadata", lambda api_key: _FakeSupadataClient(content))


def test_no_api_key_returns_stable_error(monkeypatch):
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "")
    out = yt.youtube_transcript("dQw4w9WgXcQ")
    assert out.startswith("Error: youtube_transcript disabled")


def test_empty_transcript_returns_marker(monkeypatch):
    _patch(monkeypatch, content="")
    out = yt.youtube_transcript("dQw4w9WgXcQ", raw=True)
    assert "(no transcript available)" in out


def test_raw_mode_returns_text_directly(monkeypatch):
    _patch(monkeypatch, content="some transcript text")
    out = yt.youtube_transcript("dQw4w9WgXcQ", raw=True)
    assert "some transcript text" in out


def test_raw_mode_pagination_note(monkeypatch):
    big = "A" * (yt._RAW_CHAR_LIMIT * 3)
    _patch(monkeypatch, content=big)
    out = yt.youtube_transcript("dQw4w9WgXcQ", raw=True)
    assert "A" * 100 in out
    assert f"offset={yt._RAW_CHAR_LIMIT}" in out
    assert "call youtube_transcript with offset=" in out


def test_raw_mode_offset_arithmetic(monkeypatch):
    big = "B" * (yt._RAW_CHAR_LIMIT * 3)
    _patch(monkeypatch, content=big)
    out = yt.youtube_transcript("dQw4w9WgXcQ", raw=True, offset=yt._RAW_CHAR_LIMIT)
    assert f"offset={yt._RAW_CHAR_LIMIT * 2}" in out


def test_raw_mode_no_note_below_limit(monkeypatch):
    _patch(monkeypatch, content="short transcript")
    out = yt.youtube_transcript("dQw4w9WgXcQ", raw=True)
    assert "call youtube_transcript" not in out


def test_summarizer_called_with_prompt(monkeypatch):
    _patch(monkeypatch, content="long transcript text")
    calls: list[tuple] = []

    def fake_summarize(system, user):
        calls.append((system, user))
        return "summarized result"

    monkeypatch.setattr(yt, "summarize_gemma", fake_summarize)
    out = yt.youtube_transcript("dQw4w9WgXcQ", prompt="summarize key points")
    assert out == "summarized result"
    assert calls
    assert "summarize key points" in calls[0][1]


def test_summarizer_failure_falls_back_to_raw(monkeypatch):
    _patch(monkeypatch, content="raw fallback text")

    def boom(system, user):
        raise RuntimeError("summarizer down")

    monkeypatch.setattr(yt, "summarize_gemma", boom)
    out = yt.youtube_transcript("dQw4w9WgXcQ", prompt="summarize")
    assert "raw fallback text" in out


def test_supadata_error_returns_string(monkeypatch):
    from supadata.errors import SupadataError
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "test-key")

    def _raise(api_key):
        class _Bad:
            def metadata(self, **kw):
                raise SupadataError(error="not-found", message="Video not found", details="")
            def transcript(self, **kw):
                raise SupadataError(error="not-found", message="Video not found", details="")
        return _Bad()

    monkeypatch.setattr(yt, "Supadata", _raise)
    out = yt.youtube_transcript("dQw4w9WgXcQ", raw=True)
    assert "Error:" in out


def test_schema_raw_param_uses_constant():
    (schema,) = yt.SCHEMAS
    raw_desc = schema["function"]["parameters"]["properties"]["raw"]["description"]
    assert f"{yt._RAW_CHAR_LIMIT:,}" in raw_desc
