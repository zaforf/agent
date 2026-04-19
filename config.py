import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
# Tier-1 / primary — agent loop (`agent._clients`), full user + tool context.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def gemini_summarizer_api_key() -> str:
    """API key for Gemma 26B summarizer HTTP calls only (`summarizer.summarize_gemma`).

    Reads the environment at call time so tests can monkeypatch `os.environ`.

    Prefer ``GEMINI_SUMMARIZER_API_KEY`` or ``GEMINI_API_KEY_FREE``; if both are
    unset, falls back to ``GEMINI_API_KEY`` with a log warning.
    """
    free = os.environ.get("GEMINI_SUMMARIZER_API_KEY", "") or os.environ.get(
        "GEMINI_API_KEY_FREE", ""
    )
    if free:
        return free
    tier1 = os.environ.get("GEMINI_API_KEY", "")
    if tier1:
        log.warning(
            "GEMINI_SUMMARIZER_API_KEY / GEMINI_API_KEY_FREE unset — using GEMINI_API_KEY "
            "for summarizer (fetch_url, history). Set a dedicated free-tier key to avoid "
            "sharing tier-1 quota with summarization."
        )
        return tier1
    return ""

# Provider chain — tried in order on rate-limit/failure.
#
#   1. Gemma 4 31B  (dense, #3 open model globally, strong reasoning + tool use)
#   2. Gemma 4 26B  (MoE, 3.8B active params — extremely fast, same quality tier)
#   3. Cerebras Q3  (Qwen3-235B, 1M tok/day free, strong fallback)
#   4. Groq L3.3    (last resort — proven reliable, no daily cap)
#
# Gemini key is sent as ?key= query param (not Bearer) — required for
# the new AI Studio key format (AQ. prefix).
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

PROVIDERS: list[dict] = [
    {
        "name":     "gemini-gemma4-31b",
        "api_key":  GEMINI_API_KEY,
        "base_url": GEMINI_BASE_URL,
        "model":    "gemma-4-31b-it",
    },
    {
        "name":     "gemini-gemma4-26b",
        "api_key":  GEMINI_API_KEY,
        "base_url": GEMINI_BASE_URL,
        "model":    "gemma-4-26b-a4b-it",
    },
    {
        "name":     "cerebras",
        "api_key":  CEREBRAS_API_KEY,
        "base_url": "https://api.cerebras.ai/v1",
        "model":    "qwen-3-235b-a22b-instruct-2507",
    },
    {
        "name":     "groq",
        "api_key":  GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama-3.3-70b-versatile",
    },
]

DATA_DIR           = Path(__file__).parent / "data"
SYSTEM_PROMPT_PATH = DATA_DIR / "system_prompt.md"

# Memory (Mem0 + Qdrant). Host is configurable so dev/prod can differ
# (dev: localhost; prod docker-compose: `qdrant`).
QDRANT_HOST  = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT  = int(os.environ.get("QDRANT_PORT", "6333"))
MEM0_USER_ID = "user"
