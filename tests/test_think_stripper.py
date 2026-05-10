"""Unit tests for the streaming thinking-tag stripper.

_ThinkStripper is a tight state machine (scanning → buffering → scanning, loops)
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
    assert state == "scanning"


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


def test_scanning_resumes_after_close():
    # After the closing tag the stripper returns to scanning, not a terminal
    # passthrough state, so subsequent chunks are scanned for new blocks.
    s = agent._ThinkStripper()
    s.feed("<thought>x</thought>")
    assert s._state == "scanning"
    assert s.feed("abc") == "abc"
    assert s.feed("<not-a-tag>") == "<not-a-tag>"


def test_close_tag_is_loose_by_design():
    # DESIGN: the stripper treats any of </thought>, </think>, </thinking> as a
    # closer regardless of which opener was seen. Models occasionally emit
    # mismatched tags; being forgiving here keeps visible text flowing.
    out, state = _drive(["<thought>a</thinking>b"])
    assert out == "b"
    assert state == "scanning"


def test_multiple_thinking_blocks_all_stripped():
    # Model emits: think → text → think → text. Both blocks must be stripped,
    # both visible segments must pass through (issue #46 bug 2).
    out, state = _drive([
        "<thinking>first block</thinking>",
        "visible one ",
        "<thinking>second block</thinking>",
        "visible two",
    ])
    assert out == "visible one visible two"
    assert "first block" not in out
    assert "second block" not in out
    assert state == "scanning"


def test_thinking_chars_emitted_for_second_block():
    # Verify thinking_chars events fire for a second thinking block that follows
    # visible output — the core of issue #46 bug 2.
    # We test the _ThinkStripper directly since that's where the fix lives.
    s = agent._ThinkStripper()
    events = []

    def _feed_and_record(chunk):
        forwarded = s.feed(chunk)
        if forwarded:
            events.append(("text", forwarded))
        elif s._state == "buffering":
            events.append(("thinking", len(s._buf)))

    _feed_and_record("<thinking>block one</thinking>")
    _feed_and_record("visible text ")
    _feed_and_record("<thinking>block two")   # mid-block, not yet closed
    _feed_and_record("</thinking>")
    _feed_and_record("more visible")

    thinking_events = [e for e in events if e[0] == "thinking"]
    text_events     = [e for e in events if e[0] == "text"]

    assert thinking_events, "thinking_chars must fire for the second block"
    assert any("visible text" in t for _, t in text_events)
    assert any("more visible" in t for _, t in text_events)


# ── Code-span protection ──────────────────────────────────────────────────────

def test_thinking_tag_inside_code_span_passes_through():
    # A literal `<thinking>` inside a backtick code span is visible text, not
    # a reasoning block.  The primary failing case: issue titles like
    # "model cut short when printing `<thinking>`".
    out, state = _drive(["`<thinking>`"])
    assert out == "`<thinking>`"
    assert state == "scanning"


def test_thinking_tag_inside_code_span_in_table_row():
    row = "| 98 | model output cut short printing `<thinking>` | | 2026-05-07 |\n"
    out, state = _drive([row])
    assert out == row
    assert state == "scanning"


def test_thinking_tag_inside_code_span_split_across_chunks():
    # Backtick in one chunk, tag content in next — parity tracks across boundary.
    out, state = _drive([
        "before `",
        "<thinking>literal</thinking>",
        "` after",
    ])
    assert out == "before `<thinking>literal</thinking>` after"
    assert state == "scanning"


def test_real_thinking_tag_outside_code_span_still_stripped():
    # A real thinking block outside any code span is still removed.
    out, state = _drive(["`code`", " text ", "<thinking>hidden</thinking>", " visible"])
    assert "hidden" not in out
    assert "visible" in out
    assert "`code`" in out
    assert state == "scanning"


def test_code_span_then_real_thinking_block():
    # Tag inside span is literal; tag outside span is stripped.
    out, _ = _drive(["`<thinking>` prose <thinking>real</thinking> end"])
    assert "`<thinking>`" in out
    assert "real" not in out
    assert "prose" in out
    assert "end" in out


def test_partial_thinking_tag_inside_code_span_not_held():
    # A partial tag at the end of a buffer is normally held back to wait for
    # the next chunk — but if it's inside a code span it should be released.
    s = agent._ThinkStripper()
    out = s.feed("`<thi")
    # Inside code span → should NOT hold back the partial tag
    assert "`<thi" in out


def test_partial_thinking_tag_outside_code_span_still_held():
    # Outside a code span, a partial tag at the end must still be held back.
    s = agent._ThinkStripper()
    out = s.feed("text <thi")
    assert out == "text "   # partial tag held back
    assert s._buf == "<thi"

