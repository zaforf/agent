import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CEREBRAS_API_KEY   = os.environ.get("CEREBRAS_API_KEY", "")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")

# Provider chain — tried in order on rate-limit/failure.
#
#   1. Gemma 4 31B  (dense, #3 open model globally, strong reasoning + tool use)
#   2. Gemma 4 26B  (MoE, 3.8B active params — extremely fast, same quality tier)
#   3. Cerebras Q3  (Qwen3-235B, 1M tok/day free, strong fallback)
#   4. Groq L3.3    (last resort — proven reliable, no daily cap)
#
# Both Gemma entries share GEMINI_API_KEY but different model IDs.
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

# Back-compat shims
INFERENCE_MODEL    = PROVIDERS[0]["model"]
INFERENCE_BASE_URL = PROVIDERS[0]["base_url"]
INFERENCE_API_KEY  = PROVIDERS[0]["api_key"]
FALLBACK_MODEL     = PROVIDERS[-1]["model"]
FALLBACK_BASE_URL  = PROVIDERS[-1]["base_url"]
FALLBACK_API_KEY   = PROVIDERS[-1]["api_key"]

# Paths
DATA_DIR           = Path(__file__).parent / "data"
SYSTEM_PROMPT_PATH = DATA_DIR / "system_prompt.md"

# Memory (Mem0 + Qdrant)
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
MEM0_USER_ID = "user"
