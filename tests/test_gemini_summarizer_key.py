"""Dual Gemini keys: summarizer uses free-tier env when set (DESIGN §2, issue #28)."""
from __future__ import annotations

import logging

import config
import summarizer


def test_summarizer_prefers_gemini_summarizer_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "tier1-secret")
    monkeypatch.setenv("GEMINI_SUMMARIZER_API_KEY", "free-secret")
    monkeypatch.delenv("GEMINI_API_KEY_FREE", raising=False)
    assert config.gemini_summarizer_api_key() == "free-secret"


def test_summarizer_accepts_gemini_api_key_free_alias(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "tier1-secret")
    monkeypatch.delenv("GEMINI_SUMMARIZER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_FREE", "free-alt")
    assert config.gemini_summarizer_api_key() == "free-alt"


def test_summarizer_falls_back_to_tier1_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "tier1-only")
    monkeypatch.delenv("GEMINI_SUMMARIZER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FREE", raising=False)
    caplog.set_level(logging.WARNING)
    assert config.gemini_summarizer_api_key() == "tier1-only"
    assert any("GEMINI_API_KEY" in r.message for r in caplog.records)


def test_summarizer_empty_when_no_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_SUMMARIZER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FREE", raising=False)
    assert config.gemini_summarizer_api_key() == ""


def test_summarize_gemma_posts_with_resolved_key(monkeypatch):
    """summarize_gemma uses gemini_summarizer_api_key() for the ?key= param."""
    captured: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "ok"}]}},
                ],
            }

    def _fake_post(url, params=None, json=None):
        captured["key"] = params.get("key") if params else None
        return _Resp()

    monkeypatch.setenv("GEMINI_API_KEY", "k-main")
    monkeypatch.setenv("GEMINI_SUMMARIZER_API_KEY", "k-free")

    import httpx

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, params=None, json=None):
            return _fake_post(url, params=params, json=json)

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    out = summarizer.summarize_gemma("sys", "user")
    assert out == "ok"
    assert captured["key"] == "k-free"
