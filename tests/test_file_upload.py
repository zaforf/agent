"""Tests for file upload endpoint and multimodal chat contract.

Covers:
- POST /upload: size rejection (413), MIME rejection (415), PDF extraction (422 on corrupt)
- POST /upload: valid text/plain accepted (belt-and-suspenders; primary path is PDF)
- _build_user_content: plain-string passthrough (backward compat), text attachment,
  image attachment, empty message with attachment
- /chat/stream with attachments: agent receives multimodal content list
- History round-trip: list content stored and retrieved correctly
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import agent
import db
import main
from main import Attachment, _build_user_content


# ── /upload endpoint ──────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_db, monkeypatch):
    monkeypatch.setattr(main, "_cache", {})
    with TestClient(main.app) as c:
        yield c


def test_upload_rejects_oversize(client):
    big = b"x" * (5 * 1024 * 1024 + 1)
    r = client.post("/upload", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413
    assert "5 MB" in r.json()["detail"]


def test_upload_rejects_non_pdf_mime(client):
    r = client.post("/upload", files={"file": ("img.png", b"\x89PNG", "image/png")})
    assert r.status_code == 415


def test_upload_rejects_plain_text_mime(client):
    r = client.post("/upload", files={"file": ("note.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_upload_pdf_extracts_text(client, monkeypatch):
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "extracted content"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch("main.PdfReader", return_value=fake_reader, create=True):
        # Patch pypdf import inside the endpoint
        import sys
        fake_pypdf = MagicMock()
        fake_pypdf.PdfReader = lambda f: fake_reader
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        minimal_pdf = b"%PDF-1.4 fake"
        r = client.post("/upload", files={"file": ("doc.pdf", minimal_pdf, "application/pdf")})

    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "text"
    assert body["filename"] == "doc.pdf"
    assert "extracted content" in body["content"]


def test_upload_corrupt_pdf_returns_422(client, monkeypatch):
    import sys
    fake_pypdf = MagicMock()
    fake_pypdf.PdfReader.side_effect = Exception("bad pdf")
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    r = client.post("/upload", files={"file": ("bad.pdf", b"notapdf", "application/pdf")})
    assert r.status_code == 422
    assert "extraction failed" in r.json()["detail"]


# ── _build_user_content ───────────────────────────────────────────────────────

def test_build_no_attachments_returns_string():
    result = _build_user_content("hello", [])
    assert result == "hello"


def test_build_text_attachment_produces_list():
    att = Attachment(type="text", filename="foo.py", content="def main(): pass")
    result = _build_user_content("explain this", [att])
    assert isinstance(result, list)
    texts = [p["text"] for p in result if p.get("type") == "text"]
    assert any("foo.py" in t for t in texts)
    assert any("def main" in t for t in texts)
    assert any("explain this" in t for t in texts)


def test_build_image_attachment_produces_image_url_part():
    att = Attachment(type="image", filename="shot.png", content="data:image/png;base64,abc123")
    result = _build_user_content("describe", [att])
    assert isinstance(result, list)
    img_parts = [p for p in result if p.get("type") == "image_url"]
    assert len(img_parts) == 1
    assert img_parts[0]["image_url"]["url"] == "data:image/png;base64,abc123"


def test_build_empty_message_with_attachment():
    att = Attachment(type="text", filename="notes.txt", content="some notes")
    result = _build_user_content("", [att])
    assert isinstance(result, list)
    # No empty text part appended
    text_parts = [p for p in result if p.get("type") == "text"]
    assert all(p["text"] for p in text_parts)


def test_build_files_before_message():
    att = Attachment(type="text", filename="a.txt", content="file content")
    result = _build_user_content("user question", [att])
    types = [p["type"] for p in result]
    # File text part appears before the user message text part
    assert types.index("text") < len(types) - 1
    assert result[-1]["text"] == "user question"


# ── /chat/stream with attachments ────────────────────────────────────────────

def test_chat_stream_text_attachment_reaches_agent(client, monkeypatch):
    received = {}

    async def fake_stream(user_content, history, **kwargs):
        received["content"] = user_content
        yield {"type": "text_chunk", "text": "ok"}
        yield {"type": "done", "provider": "fake", "turn_messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "ok"},
        ]}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream("POST", "/chat/stream", json={
        "message": "explain it",
        "session_id": "att-txt",
        "attachments": [{"type": "text", "filename": "code.py", "content": "def f(): pass"}],
    }) as r:
        assert r.status_code == 200
        list(r.iter_lines())

    content = received["content"]
    assert isinstance(content, list), "multimodal content must be a list when attachments present"
    texts = [p["text"] for p in content if p.get("type") == "text"]
    assert any("code.py" in t for t in texts)
    assert any("def f" in t for t in texts)
    assert any("explain it" in t for t in texts)


def test_chat_stream_image_attachment_passes_image_url(client, monkeypatch):
    received = {}

    async def fake_stream(user_content, history, **kwargs):
        received["content"] = user_content
        yield {"type": "text_chunk", "text": "ok"}
        yield {"type": "done", "provider": "fake", "turn_messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "ok"},
        ]}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream("POST", "/chat/stream", json={
        "message": "describe",
        "session_id": "att-img",
        "attachments": [{"type": "image", "filename": "sc.png", "content": "data:image/png;base64,XYZ"}],
    }) as r:
        assert r.status_code == 200
        list(r.iter_lines())

    img_parts = [p for p in received["content"] if p.get("type") == "image_url"]
    assert img_parts[0]["image_url"]["url"] == "data:image/png;base64,XYZ"


def test_chat_stream_patches_display_files_into_history(client, monkeypatch):
    """_display_files is patched onto the user message so display history can
    render file chips after a page refresh (DESIGN §13.4)."""
    async def fake_stream(user_content, history, **kwargs):
        yield {"type": "text_chunk", "text": "ok"}
        yield {"type": "done", "provider": "fake", "turn_messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "ok"},
        ]}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream("POST", "/chat/stream", json={
        "message": "look at this",
        "session_id": "df-stream",
        "attachments": [{"type": "image", "filename": "photo.jpg", "content": "data:image/jpeg;base64,xyz"}],
    }) as r:
        assert r.status_code == 200
        list(r.iter_lines())

    msgs = db.get_display_history("df-stream")
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert user_msg.get("attachments") == [{"type": "image", "filename": "photo.jpg"}]
    assert user_msg["content"] == "look at this"


def test_chat_stream_no_attachments_backward_compat(client, monkeypatch):
    received = {}

    async def fake_stream(user_content, history, **kwargs):
        received["content"] = user_content
        yield {"type": "text_chunk", "text": "hi"}
        yield {"type": "done", "provider": "fake", "turn_messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "hi"},
        ]}

    monkeypatch.setattr(agent, "run_stream", fake_stream)

    with client.stream("POST", "/chat/stream",
                       json={"message": "hello", "session_id": "noatt"}) as r:
        assert r.status_code == 200
        list(r.iter_lines())

    assert received["content"] == "hello", "plain message without attachments must stay a string"


# ── History round-trip with multimodal content ────────────────────────────────

def test_display_history_shows_message_text_and_chip_metadata(tmp_db):
    """Display history shows the user's typed message text (not file contents) and
    an 'attachments' list for chip rendering in the UI (DESIGN §13.4)."""
    multimodal_content = [
        {"type": "text", "text": "[File: readme.md]\nsome notes"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "summarise"},
    ]
    display_files = [
        {"type": "text",  "filename": "readme.md"},
        {"type": "image", "filename": "shot.png"},
    ]
    turn = [
        {"role": "user", "content": multimodal_content, "_display_files": display_files},
        {"role": "assistant", "content": "here is the summary"},
    ]
    db.append_turn("hist-multi", "summarise", turn)

    msgs = db.get_display_history("hist-multi")
    user_msg = next(m for m in msgs if m["role"] == "user")

    # Content shows the typed message, not the file contents
    assert user_msg["content"] == "summarise"
    assert "some notes" not in user_msg["content"], "file body must not appear in display text"

    # Chip metadata is present for the UI
    assert user_msg["attachments"] == display_files


def test_display_history_no_attachments_field_when_none(tmp_db):
    """User messages without files must not have an 'attachments' key."""
    turn = [
        {"role": "user",      "content": "plain message"},
        {"role": "assistant", "content": "reply"},
    ]
    db.append_turn("hist-plain", "plain message", turn)

    msgs = db.get_display_history("hist-plain")
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert "attachments" not in user_msg


def test_get_history_preserves_multimodal_content_for_llm(tmp_db):
    """The LLM-facing get_history() must return the original list intact."""
    multimodal_content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xyz"}},
    ]
    turn = [
        {"role": "user",      "content": multimodal_content},
        {"role": "assistant", "content": "response"},
    ]
    db.append_turn("hist-raw", "hello", turn)

    msgs = db.get_history("hist-raw")
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert isinstance(user_msg["content"], list), "LLM history must preserve raw list content"
    assert user_msg["content"] == multimodal_content
