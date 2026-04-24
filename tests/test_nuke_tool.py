from __future__ import annotations

import json

import pytest

from tools import nuke


def test_nuke_chat_returns_marker_payload():
    out = nuke.nuke_chat("Carry these facts forward")
    assert out.startswith(nuke._NUKE_PREFIX)
    payload = json.loads(out[len(nuke._NUKE_PREFIX):])
    assert payload["summary"] == "Carry these facts forward"


def test_nuke_chat_requires_non_empty_summary():
    with pytest.raises(ValueError, match="non-empty"):
        nuke.nuke_chat("   ")
