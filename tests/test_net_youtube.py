"""Real-network tests for tools/youtube.py.

Marked @pytest.mark.net — skipped in default pytest run.
Run with: pytest -m net

Tests fetch the transcript of a stable, well-known video to verify the
Supadata API key, network path, and text-extraction are all wired correctly.
Uses Rick Astley — "Never Gonna Give You Up" (dQw4w9WgXcQ), which has been
on YouTube since 2009 and reliably has an English transcript.
"""
from __future__ import annotations

import pytest

import config
from tools import youtube as yt

pytestmark = pytest.mark.net

_TEST_VIDEO_ID  = "dQw4w9WgXcQ"   # Rick Astley — Never Gonna Give You Up
_TEST_VIDEO_URL = f"https://www.youtube.com/watch?v={_TEST_VIDEO_ID}"


@pytest.mark.skipif(not config.SUPADATA_API_KEY, reason="SUPADATA_API_KEY not set")
def test_fetch_transcript_by_video_id():
    """Bare video ID → resolved to URL → transcript fetched and returned as text."""
    out = yt.youtube_transcript(_TEST_VIDEO_ID, raw=True)
    assert not out.startswith("Error:"), out
    assert len(out) > 100, f"transcript suspiciously short: {out!r}"
    # Should not contain raw HTML or timestamp artefacts
    assert "<" not in out, "unexpected HTML in transcript"


@pytest.mark.skipif(not config.SUPADATA_API_KEY, reason="SUPADATA_API_KEY not set")
def test_fetch_transcript_by_full_url():
    """Full YouTube URL also resolves correctly."""
    out = yt.youtube_transcript(_TEST_VIDEO_URL, raw=True)
    assert not out.startswith("Error:"), out
    assert len(out) > 100


@pytest.mark.skipif(not config.SUPADATA_API_KEY, reason="SUPADATA_API_KEY not set")
def test_fetch_transcript_by_short_url():
    """youtu.be short URL resolves to same transcript."""
    short = f"https://youtu.be/{_TEST_VIDEO_ID}"
    out = yt.youtube_transcript(short, raw=True)
    assert not out.startswith("Error:"), out
    assert len(out) > 100
