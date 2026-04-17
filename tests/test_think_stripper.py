"""Unit tests for the streaming thinking-tag stripper.

_ThinkStripper is a tight state machine (scanning → buffering → passthrough)
that must never leak an open thinking tag to the user and never drop visible
content. These tests drive it chunk-by-chunk so chunk boundary bugs surface.
"""
from __future__ import annotations

import agent


def _drive(chunks):
    """Feed chunks sequentially and return (visible_output, final_state)."""
    s = agent._ThinkStripper()
    out = []
    for c in chunks:
        out.append(s.feed(c))
    tail = s.finalize()
    return "".join(out) + tail, s._state


def test_no_tags_pass_through_immediately():
    out, state = _drive(["hello ", "world"])
    assert out == "hello world"
    assert state == "scanning"


def test_single_chunk_with_block_stripped():
    out, _ = _drive(["<thought>a</thought>hi"])
    assert out == "hi"


def test_block_split_across_chunks():
    out, state = _drive(["<thou", "ght>secret", "</thought>", "visible"])
    assert out == "visible"
    assert state == "passthrough"


def test_visible_before_block_is_emitted():
    out, _ = _drive(["pre ", "<thought>x</thought>", "post"])
    assert out == "pre post"


def test_partial_open_tag_does_not_leak():
    # '<thi' is ambiguous — could become <thinking> — so stripper must hold it
    # back until it disambiguates.
    s = agent._ThinkStripper()
    forwarded = s.feed("hello <thi")
    assert forwarded == "hello "
    forwarded2 = s.feed("nking>secret</thinking>bye")
    assert forwarded2 == "bye"


# Note: an unclosed `<thought>` block is a model malfunction. We don't pin a
# specific recovery here — the agent-loop repair pass (covered in
# test_agent_loop) is the real safety net when visible content is empty.


def test_no_tag_but_partial_prefix_finalized():
    # If the stream ends with something that looks like it might be an opening
    # tag (e.g. just "<") but never becomes one, the visible prefix must be
    # emitted (during feed() or on finalize).
    s = agent._ThinkStripper()
    emitted = s.feed("answer is <")
    tail = s.finalize()
    full = emitted + tail
    assert "answer is" in full, f"visible prefix dropped: emitted={emitted!r} tail={tail!r}"


def test_passthrough_after_close_is_cheap():
    s = agent._ThinkStripper()
    s.feed("<thought>x</thought>")
    # Now in passthrough — subsequent chunks returned verbatim without buffering
    assert s.feed("abc") == "abc"
    assert s.feed("<not-a-tag>") == "<not-a-tag>"


def test_close_tag_is_loose_by_design():
    # DESIGN: the stripper treats any of </thought>, </think>, </thinking> as a
    # closer regardless of which opener was seen. Models occasionally emit
    # mismatched tags; being forgiving here keeps visible text flowing.
    out, state = _drive(["<thought>a</thinking>b"])
    assert out == "b"
    assert state == "passthrough"

