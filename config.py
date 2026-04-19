import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def gemini_free_api_key() -> str:
    """Gemma 26B summarizer (`summarize_gemma` only). Reads env at call time for tests."""
    if k := os.environ.get("GEMINI_API_KEY_FREE", ""):
        return k
    if k := os.environ.get("GEMINI_API_KEY", ""):
        log.warning("GEMINI_API_KEY_FREE unset; using GEMINI_API_KEY for summarizer")
        return k
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
