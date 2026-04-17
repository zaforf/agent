"""Tests for gemini_client format translation.

The Gemini native API takes a different shape than OpenAI's. `gemini_client.py`
translates between them. These tests lock the translation contract so a future
refactor can't silently corrupt tool-call round-trips.
"""
from __future__ import annotations

import json

import gemini_client as gc


def test_system_becomes_system_instruction():
    contents, sys_inst = gc._to_gemini_contents([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ])
    assert sys_inst == {"parts": [{"text": "You are helpful."}]}
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_user_message_simple():
    contents, _ = gc._to_gemini_contents([{"role": "user", "content": "hello"}])
    assert contents == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_assistant_with_tool_calls_emits_function_call_parts():
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "tc_1",
            "type": "function",
            "function": {"name": "recall", "arguments": json.dumps({"query": "foo"})},
        }],
    }]
    contents, _ = gc._to_gemini_contents(msgs)
    assert contents[0]["role"] == "model"
    fc_parts = [p for p in contents[0]["parts"] if "functionCall" in p]
    assert len(fc_parts) == 1
    assert fc_parts[0]["functionCall"]["name"] == "recall"
    assert fc_parts[0]["functionCall"]["args"] == {"query": "foo"}


def test_tool_response_grouped_as_user_function_response():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "recall", "arguments": "{}"}},
        ]},
        {"role": "tool", "name": "recall", "tool_call_id": "a", "content": "result"},
    ]
    contents, _ = gc._to_gemini_contents(msgs)
    # First: model functionCall
    assert contents[0]["role"] == "model"
    # Second: user functionResponse
    assert contents[1]["role"] == "user"
    fr_parts = [p for p in contents[1]["parts"] if "functionResponse" in p]
    assert fr_parts[0]["functionResponse"]["name"] == "recall"
    assert fr_parts[0]["functionResponse"]["response"]["content"] == "result"


def test_consecutive_tool_responses_grouped_in_one_user_turn():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "recall", "arguments": "{}"}},
            {"id": "b", "type": "function",
             "function": {"name": "recall", "arguments": "{}"}},
        ]},
        {"role": "tool", "name": "recall", "tool_call_id": "a", "content": "r1"},
        {"role": "tool", "name": "recall", "tool_call_id": "b", "content": "r2"},
    ]
    contents, _ = gc._to_gemini_contents(msgs)
    # The two tool messages must collapse into a single user turn with two
    # functionResponse parts — Gemini rejects interleaved tool turns otherwise.
    user_turns = [c for c in contents if c["role"] == "user"]
    assert len(user_turns) == 1
    fr_parts = [p for p in user_turns[0]["parts"] if "functionResponse" in p]
    assert len(fr_parts) == 2


def test_to_gemini_tools_shape():
    tools = [{
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search memory",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }]
    out = gc._to_gemini_tools(tools)
    assert out == [{
        "functionDeclarations": [{
            "name": "recall",
            "description": "Search memory",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }],
    }]


def test_parse_candidate_extracts_visible_text_only():
    candidate = {"content": {"parts": [
        {"text": "hello "},
        {"text": "world"},
    ]}}
    choice = gc._parse_candidate(candidate)
    assert choice.message.content == "hello world"
    assert choice.message.tool_calls == []


def test_parse_candidate_skips_thought_parts():
    candidate = {"content": {"parts": [
        {"text": "secret scratch", "thought": True},
        {"text": "visible reply"},
    ]}}
    choice = gc._parse_candidate(candidate)
    assert choice.message.content == "visible reply", (
        "thought parts must be excluded from visible content"
    )


def test_parse_candidate_emits_tool_calls():
    candidate = {"content": {"parts": [
        {"functionCall": {"name": "recall", "args": {"query": "zafir"}}},
    ]}}
    choice = gc._parse_candidate(candidate)
    tcs = choice.message.tool_calls
    assert len(tcs) == 1
    assert tcs[0].function.name == "recall"
    assert json.loads(tcs[0].function.arguments) == {"query": "zafir"}
    assert tcs[0].id.startswith("tc_"), tcs[0].id
