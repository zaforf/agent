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
SUPADATA_API_KEY      = os.environ.get("SUPADATA_API_KEY", "")
BRAVE_SEARCH_QUOTA_COOLDOWN_S = float(os.environ.get("BRAVE_SEARCH_QUOTA_COOLDOWN_S", "300"))

# Provider model IDs are configurable so a model retirement does not require
# editing several modules independently.
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Optional: Telegram bot long-polling transport (issue #60). Empty = disabled.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# Optional: restrict Telegram bot to these numeric user IDs (comma-separated).
# When unset or empty, any user may message the bot. When set, all others are ignored.
_tg_allow_raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "").strip()
if not _tg_allow_raw:
    TELEGRAM_ALLOWED_USER_IDS: frozenset[int] | None = None
else:
    _ids: list[int] = []
    for _part in _tg_allow_raw.split(","):
        _p = _part.strip()
        if not _p:
            continue
        try:
            _ids.append(int(_p))
        except ValueError:
            log.warning("TELEGRAM_ALLOWED_USER_IDS: skip invalid segment %r", _p)
    TELEGRAM_ALLOWED_USER_IDS = frozenset(_ids) if _ids else None

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
#   3. Cerebras GPT-OSS 120B (strong fallback; account quota dependent)
#   4. Groq GPT-OSS 120B     (last resort; free-tier limits apply)
#
# All Gemini calls go through the OpenAI-compat endpoint via AsyncOpenAI.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEBUG_LOGGING = os.environ.get("DEBUG_LOGGING", "false").lower() == "true"

# Hint to the chat API that the model may emit multiple tool_calls in one assistant
# message; the server runs read-only batches concurrently (see agent.py).
AGENT_PARALLEL_TOOL_CALLS = os.environ.get("AGENT_PARALLEL_TOOL_CALLS", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Provider policy. ``responsive`` is the low-development-time experiment: use
# the fast Gemma 26B path for every request and retain the rest as fallbacks.
# ``quality`` enables the older conservative task-aware ordering for comparison.
_provider_mode_raw = os.environ.get("AGENT_PROVIDER_MODE", "responsive").strip().lower()
if _provider_mode_raw not in {"responsive", "quality"}:
    log.warning("AGENT_PROVIDER_MODE=%r is invalid; using responsive", _provider_mode_raw)
    _provider_mode_raw = "responsive"
AGENT_PROVIDER_MODE = _provider_mode_raw

# In quality mode, deep/ambiguous requests retain the quality-first chain;
# obvious quick/current/inspection requests can use a faster provider without
# changing the fallback set.
AGENT_PROVIDER_ROUTING_ENABLED = os.environ.get("AGENT_PROVIDER_ROUTING_ENABLED", "true").lower() in (
    "1", "true", "yes",
)

# Avoid rediscovering a provider-wide quota/payment failure on every turn.
# This is a temporary circuit breaker, not a permanent provider disablement.
PROVIDER_RATE_LIMIT_COOLDOWN_S = float(os.environ.get("PROVIDER_RATE_LIMIT_COOLDOWN_S", "60"))
PROVIDER_API_ERROR_COOLDOWN_S = float(os.environ.get("PROVIDER_API_ERROR_COOLDOWN_S", "300"))
PROVIDER_STREAM_INTERRUPT_COOLDOWN_S = float(
    os.environ.get("PROVIDER_STREAM_INTERRUPT_COOLDOWN_S", "10")
)
# Emit honest SSE progress while a provider is still preparing its stream.
# This is especially useful for reasoning models whose first chunk contains
# hidden thinking and may arrive several seconds after the request begins.
PROVIDER_WAIT_STATUS_INTERVAL_S = float(
    os.environ.get("PROVIDER_WAIT_STATUS_INTERVAL_S", "2")
)

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
        "model":    CEREBRAS_MODEL,
    },
    {
        "name":     "groq",
        "api_key":  GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "model":    GROQ_MODEL,
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

# Ambient memory prefetch: run a recall automatically before the first LLM call each turn.
# Threshold is cosine similarity (0–1); results below it are suppressed as noise.
MEMORY_PREFETCH_TOP_K    = int(os.environ.get("MEMORY_PREFETCH_TOP_K", "5"))
MEMORY_PREFETCH_THRESHOLD = float(os.environ.get("MEMORY_PREFETCH_THRESHOLD", "0.3"))
# Ambient recall is an enhancement, never a prerequisite for responding. Mem0
# may cold-start by inspecting/migrating Qdrant collections, so bound it.
# Enabled by default: memory is part of the personal-agent experience. The
# timeout below keeps an unavailable/cold memory service from blocking a turn.
MEMORY_PREFETCH_ENABLED = os.environ.get("MEMORY_PREFETCH_ENABLED", "true").lower() in (
    "1", "true", "yes",
)
MEMORY_PREFETCH_TIMEOUT_S = float(os.environ.get("MEMORY_PREFETCH_TIMEOUT_S", "1.5"))
MEMORY_WARM_ON_STARTUP = os.environ.get("MEMORY_WARM_ON_STARTUP", "true").lower() in (
    "1", "true", "yes",
)
# Explicit memory operations are valuable but must not make a turn hang when
# Mem0/Qdrant or its embedding provider is unhealthy.
MEMORY_TOOL_TIMEOUT_S = float(os.environ.get("MEMORY_TOOL_TIMEOUT_S", "30"))

# After a successful ``workspace_search_replace`` on a ``.py`` file, optionally append
# ``ruff check`` / scoped ``pytest`` output to the same tool result (see ``tools/post_edit_verify.py``).
AGENT_POST_EDIT_VERIFY = os.environ.get("AGENT_POST_EDIT_VERIFY", "true").lower() in ("1", "true", "yes")
AGENT_POST_EDIT_PYTEST = os.environ.get("AGENT_POST_EDIT_PYTEST", "false").lower() in ("1", "true", "yes")
AGENT_POST_EDIT_VERIFY_TIMEOUT = int(os.environ.get("AGENT_POST_EDIT_VERIFY_TIMEOUT", "120"))
