import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

CEREBRAS_API_KEY      = os.environ.get("CEREBRAS_API_KEY", "")
GROQ_API_KEY          = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY", "")
BRAVE_SEARCH_API_KEY  = os.environ.get("BRAVE_SEARCH_API_KEY", "")

_free = os.environ.get("GEMINI_API_KEY_FREE", "")
if _free:
    GEMINI_API_KEY_FREE_RESOLVED = _free
elif GEMINI_API_KEY:
    GEMINI_API_KEY_FREE_RESOLVED = GEMINI_API_KEY
    log.warning("GEMINI_API_KEY_FREE unset; using GEMINI_API_KEY for summarizer")
else:
    GEMINI_API_KEY_FREE_RESOLVED = ""

# Provider chain — tried in order on rate-limit/failure.
#
#   1. Gemma 4 31B  (dense, #3 open model globally, strong reasoning + tool use)
#   2. Gemma 4 26B  (MoE, 3.8B active params — extremely fast, same quality tier)
#   3. Cerebras Q3  (Qwen3-235B, 1M tok/day free, strong fallback)
#   4. Groq L3.3    (last resort — proven reliable, no daily cap)
#
# All Gemini calls go through the OpenAI-compat endpoint via AsyncOpenAI.
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

# Mem0 embedder: Gemini API (`google-genai`), same key as chat (`GEMINI_API_KEY`).
# Collection name bumped when embedding model/dims change (no migration of old vectors).
MEM0_QDRANT_COLLECTION = os.environ.get("MEM0_QDRANT_COLLECTION", "agent_memories_gemini")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GEMINI_EMBEDDING_DIMS = int(os.environ.get("GEMINI_EMBEDDING_DIMS", "768"))

# Mem0 LLM (fact extraction on remember, etc.) — native Gemini API via mem0, not Groq.
# Default: Gemma 4 26B MoE (lighter than 31B; avoids Groq free-tier TPM limits on large prompts).
MEM0_LLM_MODEL = os.environ.get("MEM0_LLM_MODEL", "gemma-4-26b-a4b-it")
