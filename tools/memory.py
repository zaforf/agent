from mem0 import Memory

import config


def mem0_config_dict() -> dict:
    """Mem0 configuration (embeddings + vector store + LLM). Exposed for tests."""
    return {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": config.MEM0_LLM_MODEL,
                "api_key": config.GEMINI_API_KEY or None,
                "max_tokens": 2048,
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": config.GEMINI_EMBEDDING_MODEL,
                "embedding_dims": config.GEMINI_EMBEDDING_DIMS,
                "api_key": config.GEMINI_API_KEY or None,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": config.QDRANT_HOST,
                "port": config.QDRANT_PORT,
                "collection_name": config.MEM0_QDRANT_COLLECTION,
                "embedding_model_dims": config.GEMINI_EMBEDDING_DIMS,
            },
        },
    }


USER_ID = "user"
_memory = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory.from_config(mem0_config_dict())
    return _memory


def remember(content: str, category: str = "fact") -> str:
    # infer=False: embed and persist the string as-is. With infer=True (mem0 default),
    # LLM extraction can fail or return no facts — then nothing is written to Qdrant
    # but the tool still looked "successful", so recall/list stay empty.
    _get_memory().add(
        content,
        user_id=USER_ID,
        metadata={"category": category},
        infer=False,
    )
    return f"Stored: {content}"


def recall(query: str) -> str:
    # mem0ai >=2.0: session scope via filters=; use top_k (not limit).
    results = _get_memory().search(
        query,
        filters={"user_id": USER_ID},
        top_k=5,
    )
    entries = results.get("results", [])
    if not entries:
        return "No relevant memories found."
    return "\n".join(f"- {r['memory']}" for r in entries)


def list_memories() -> str:
    results = _get_memory().get_all(filters={"user_id": USER_ID})
    entries = results.get("results", [])
    if not entries:
        return "No memories stored."
    return "\n".join(f"[{r['id']}] ({r.get('metadata', {}).get('category','?')}) {r['memory']}" for r in entries)


def delete_memory(memory_id: str) -> str:
    _get_memory().delete(memory_id=memory_id)
    return f"Deleted memory {memory_id}"


# Exposed for API endpoints (not a tool)
def get_all() -> list[dict]:
    results = _get_memory().get_all(filters={"user_id": USER_ID})
    return results.get("results", [])


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store a durable fact to long-term memory. "
                "Prefer atomic memories: one distinct fact/preference per call unless multiple points are inseparable. "
                "ONLY call for facts worth recalling in a completely different future conversation: "
                "user's name, skills, ongoing projects, strong preferences, important context. "
                "Do NOT store: what was asked in this conversation, temporary context, "
                "things already in memory, or trivial facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The durable fact to store"},
                    "category": {
                        "type": "string",
                        "enum": ["user", "preference", "fact", "project"],
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search long-term memory for relevant context before answering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List all stored memories. Use to check what you know before adding duplicates.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Delete a specific memory by its ID (first 8 chars from list_memories).",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Full memory ID to delete"}
                },
                "required": ["memory_id"],
            },
        },
    },
]

FUNCTIONS = {
    "remember": remember,
    "recall": recall,
    "list_memories": list_memories,
    "delete_memory": delete_memory,
}
