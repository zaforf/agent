"""Run transparent, provider-by-provider capability checks.

This intentionally records raw outputs instead of asking another model to score
them. Review the JSON alongside ``cases.json`` when deciding model routing.

Example:
    python evals/run.py --provider groq --output /tmp/agent-eval.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evals" / "cases.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI  # noqa: E402

import agent  # noqa: E402
import config  # noqa: E402


def _provider_entries(name: str | None) -> list[dict]:
    entries = [p for p in config.PROVIDERS if p["api_key"]]
    if name:
        entries = [p for p in entries if p["name"] == name]
        if not entries:
            raise SystemExit(f"Provider not configured: {name}")
    return entries


async def _run_case(provider: dict, case: dict) -> dict:
    client = AsyncOpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
    memory = case.get("memory")
    messages = [{"role": "system", "content": agent._build_system_prompt()}]
    if memory:
        messages.append({"role": "user", "content": f"[Memory — these may help]\n- {memory}"})
    messages.append({"role": "user", "content": case["prompt"]})

    started = time.perf_counter()
    first_event = None
    first_visible = None
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    error = None
    try:
        stream = await client.chat.completions.create(
            model=provider["model"],
            messages=messages,
            tools=agent.TOOL_SCHEMAS,
            tool_choice="auto",
            stream=True,
        )
        async for chunk in stream:
            if first_event is None:
                first_event = time.perf_counter()
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                if first_visible is None:
                    first_visible = time.perf_counter()
            for tc in delta.tool_calls or []:
                tool_calls.append({
                    "index": tc.index,
                    "id": tc.id,
                    "name": getattr(tc.function, "name", None),
                    "arguments": getattr(tc.function, "arguments", None),
                })
    except Exception as exc:  # raw failure belongs in the result artifact
        error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    return {
        "provider": provider["name"],
        "model": provider["model"],
        "case": case["id"],
        "category": case["category"],
        "rubric": case["rubric"],
        "ttft_ms": round((first_event - started) * 1000, 1) if first_event else None,
        "first_visible_ms": round((first_visible - started) * 1000, 1) if first_visible else None,
        "total_ms": round((finished - started) * 1000, 1),
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "error": error,
    }


async def _run(providers: list[dict], cases: list[dict]) -> list[dict]:
    results: list[dict] = []
    for provider in providers:
        for case in cases:
            print(f"{provider['name']} / {case['id']}", flush=True)
            results.append(await _run_case(provider, case))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="configured provider name; default: every configured provider")
    parser.add_argument("--case", action="append", help="case id; repeatable")
    parser.add_argument("--output", type=Path, default=Path("eval-results.json"))
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text())
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise SystemExit(f"Unknown case(s): {', '.join(sorted(missing))}")

    results = asyncio.run(_run(_provider_entries(args.provider), cases))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
