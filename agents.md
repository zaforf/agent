# Agents working on this repo

Short orientation so dispatched agents don’t thrash.

## Source of truth

- **[DESIGN.md](DESIGN.md)** — Expected runtime behavior. If code disagrees with it, either fix the code or update `DESIGN.md` first (don’t leave them out of sync).
- **`data/system_prompt.md`** — Editable assistant persona/instructions; loaded every request via `agent._build_system_prompt()` (prepends today’s date + generated tool docs).

## Stack

- Python **3.12+** (project runs on newer 3.x as well).
- **FastAPI** entry: `main.py`. Core loop: `agent.py`. SQLite history: `db.py`. Tools registry: `tools/__init__.py`.
- LLM adapters: `gemini_client.py` (native Gemini), OpenAI-compatible clients for other providers (`config.PROVIDERS`).

## Tests

```bash
python -m venv venv && source venv/bin/activate   # once
pip install -r requirements.txt
python -m pytest                                  # default: fast, no network
```

- Default excludes `live` and `net` (see `pyproject.toml`). Those need keys / network:
  - `pytest -m live`
  - `pytest -m net`
  - Full: `pytest -m "live or net or (not live and not net)"`

After behavior changes, run the default suite at minimum. CI runs the same (`.github/workflows/ci.yml`).

## Conventions

- **Focused diffs** — Fix what you were asked to fix; don’t refactor unrelated code or add docs the user didn’t request.
- **Tools** — Add a module under `tools/` with `SCHEMAS` + `FUNCTIONS`, register in `tools/__init__.py`.
- **Secrets** — Never commit keys; use `.env` (see `.env.example` if present). Tests should stay hermetic by default.

## Files worth knowing

| Area | Files |
|------|--------|
| Agent loop / streaming | `agent.py` |
| HTTP API + SSE | `main.py` |
| History summarization (Gemma 26B) | `summarizer.py`, `_summarize_for_history` in `agent.py` |
| Web fetch | `tools/fetch.py` |
| Memory (Mem0 + Qdrant) | `tools/memory.py` |

When in doubt, search `DESIGN.md` and mirror existing patterns in the same directory.
