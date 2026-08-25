"""Web tools — `fetch_url` (retrieve and summarize a URL) and `web_search`
(Brave Search API for ranked links + snippets when the model has no URL yet).

`fetch_url` default: the full page is passed to a fast long-context summarizer
(Gemma 4 26B, 1M-token context) together with a caller-supplied prompt, so the
model receives exactly the information it asked for rather than a truncated
chunk of raw HTML text. Pass raw=True for offset-based pagination.

`web_search` is additive — it produces URLs + short snippets the model can
then `fetch_url` for depth. Free tier of the Brave Search API; key optional.
"""
import logging
import re
import time
from html.parser import HTMLParser

import httpx

from config import BRAVE_SEARCH_API_KEY
from summarizer import summarize_gemma

log = logging.getLogger(__name__)

_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "article"}

_SUMMARIZER_CHAR_LIMIT = 256_000   # chars fed to summarizer (well within model limit)
_RAW_CHAR_LIMIT        = 8_000     # chars returned in raw/paginated mode
_FETCH_RETRIES         = 4
_FETCH_TIMEOUT_S       = 15

# ── Brave Search ─────────────────────────────────────────────────────────────

_BRAVE_URL              = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_TIMEOUT_S        = 10
_SEARCH_DEFAULT_RESULTS = 10
_SEARCH_MAX_RESULTS     = 20
_SEARCH_SNIPPET_CHARS   = 240
_SEARCH_TOTAL_CHARS     = 4000


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._depth += 1
        if tag in _BLOCK_TAGS and not self._depth:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


_FETCH_SYSTEM_INSTRUCTION = (
    "You are a precise document analyst. "
    "Extract and summarize exactly what the user requests from the provided content. "
    "Be comprehensive, accurate, and well-structured."
)


def _summarize_content(text: str, prompt: str) -> str:
    """Send page text to Gemma 26B with the caller's extraction prompt."""
    if len(text) > _SUMMARIZER_CHAR_LIMIT:
        text = (
            text[:_SUMMARIZER_CHAR_LIMIT]
            + f"\n[…content truncated at {_SUMMARIZER_CHAR_LIMIT:,} chars]"
        )
    log.info("fetch-summarize  %.80s", prompt)
    return summarize_gemma(
        _FETCH_SYSTEM_INSTRUCTION,
        f"{prompt}\n\n---\n\n{text}",
    )


def _fetch_with_retries(url: str, headers: dict) -> httpx.Response:
    last_err = None
    with httpx.Client(timeout=_FETCH_TIMEOUT_S, follow_redirects=True) as client:
        for attempt in range(_FETCH_RETRIES):
            try:
                resp = client.get(url, headers=headers)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after", "").strip()
                    if retry_after.isdigit():
                        wait_s = max(1, min(20, int(retry_after)))
                    else:
                        wait_s = min(20, 2 ** attempt)
                    if attempt < _FETCH_RETRIES - 1:
                        log.warning("fetch_url: 429 from %s, retry in %ss", url, wait_s)
                        time.sleep(wait_s)
                        continue
                resp.raise_for_status()
                return resp
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as e:
                last_err = e
                if attempt < _FETCH_RETRIES - 1:
                    wait_s = min(10, 2 ** attempt)
                    time.sleep(wait_s)
                    continue
                raise
    raise RuntimeError(f"request failed: {last_err}")


def fetch_url(url: str, prompt: str = "", offset: int = 0, raw: bool = False) -> str:
    """Fetch a URL and return its content.

    With a prompt (default): passes the full page to a summarizer that extracts
    exactly what was asked. Preferred — avoids context flooding from raw text.

    With raw=True or no prompt: returns up to _RAW_CHAR_LIMIT chars starting
    from offset. Use offset pagination when you need the raw text in chunks.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = _fetch_with_retries(url, headers)

        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            parser = _TextExtractor()
            parser.feed(resp.text)
            text = parser.get_text()
        else:
            text = resp.text.strip()

        if not text:
            return "(no readable content)"

        # ── Summarizer mode (default when prompt given) ──────────────────────
        if prompt and not raw:
            try:
                return _summarize_content(text, prompt)
            except Exception as e:
                log.warning("fetch_url: summarizer failed (%s), falling back to raw text", e)

        # ── Raw / paginated mode ─────────────────────────────────────────────
        total = len(text)
        chunk = text[offset:]
        if len(chunk) > _RAW_CHAR_LIMIT:
            remaining  = total - offset - _RAW_CHAR_LIMIT
            next_offset = offset + _RAW_CHAR_LIMIT
            chunk = (
                chunk[:_RAW_CHAR_LIMIT]
                + f"\n\n[… {remaining:,} more chars — call fetch_url with offset={next_offset} to continue]"
            )
        return chunk

    except httpx.TimeoutException:
        return f"Error: request to {url} timed out ({_FETCH_TIMEOUT_S} s)"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} from {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def _strip_html(s: str) -> str:
    """Brave snippets/titles often contain `<strong>` highlight tags — drop them."""
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _clean_search_text(s: str) -> str:
    """Keep tool output API-safe for provider round-trips (strip controls/surrogates)."""
    s = _strip_html(s or "")
    s = s.encode("utf-8", "replace").decode("utf-8")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def web_search(query: str, max_results: int = _SEARCH_DEFAULT_RESULTS) -> str:
    """Brave Search — return a numbered markdown list of title, URL, snippet.

    Use to discover URLs for `fetch_url` when none is at hand. `max_results`
    is clamped to [1, _SEARCH_MAX_RESULTS]. Returns a stable error string when
    the API key is missing or the request fails (so the model can react).
    """
    if not BRAVE_SEARCH_API_KEY:
        return "Error: web_search disabled — set BRAVE_SEARCH_API_KEY in .env"

    q = (query or "").strip()
    if not q:
        return "Error: web_search needs a non-empty query"

    n = max(1, min(int(max_results or _SEARCH_DEFAULT_RESULTS), _SEARCH_MAX_RESULTS))
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }
    params = {"q": q, "count": n}

    try:
        with httpx.Client(timeout=_BRAVE_TIMEOUT_S) as client:
            resp = client.get(_BRAVE_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 402:
            # Brave uses 402 for exhausted monthly credits, distinct from a
            # transient 429 rate limit. Keep this actionable for the model/UI.
            try:
                detail = e.response.json().get("error", {}).get("detail", "")
            except ValueError:
                detail = "monthly usage limit exceeded"
            detail = detail or "monthly usage limit exceeded"
            return f"Error: web_search quota exhausted ({detail}). Configure a new Brave Search plan or API key."
        return f"Error: web_search HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        return f"Error: web_search timed out ({_BRAVE_TIMEOUT_S} s)"
    except Exception as e:
        return f"Error: web_search failed: {e}"

    results = (data.get("web") or {}).get("results") or []
    if not results:
        return f"No results for {q!r}."

    lines = []
    for i, r in enumerate(results[:n], 1):
        title = _clean_search_text(r.get("title", "")) or "(untitled)"
        url = _clean_search_text(r.get("url", ""))
        snippet = _clean_search_text(r.get("description", ""))
        if len(snippet) > _SEARCH_SNIPPET_CHARS:
            snippet = snippet[:_SEARCH_SNIPPET_CHARS].rstrip() + "…"
        lines.append(f"{i}. {title} - {url}\n   {snippet}" if snippet else f"{i}. {title} - {url}")

    out = "\n".join(lines)
    if len(out) > _SEARCH_TOTAL_CHARS:
        out = out[:_SEARCH_TOTAL_CHARS].rstrip() + "\n… [truncated]"
    return out


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a URL and extract specific information from it. "
                "Always provide a prompt describing what to extract. "
                "Note: the summarizer may prioritize conciseness; use explicit constraints "
                "(e.g. 'exhaustive', 'do not omit') when full detail is required. "
                "The full page is passed to a long-context summarizer which returns the result. "
                "Use raw=true only when you need unprocessed text (e.g. code files, data, "
                "or offset-based pagination)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What to extract or summarize from the page. "
                            "E.g. 'List all albums in chronological order with type (EP/Studio/etc)'. "
                            "Provide this whenever you want structured information from a page."
                        ),
                    },
                    "raw": {
                        "type": "boolean",
                        "description": (
                            f"Return raw text (up to {_RAW_CHAR_LIMIT:,} chars) instead of summarizing. "
                            "Use with offset for pagination. Default: false."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Character offset for raw-mode pagination (default 0). "
                            "Use the next_offset value from a truncated raw result."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    }
]

SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web (Brave Search API) for ranked links + short snippets. "
            "Use when you need a URL but don't have one yet — then call fetch_url "
            "on the most promising result for depth. Returns a numbered markdown "
            "list of title / URL / snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Number of results to return "
                        f"(default {_SEARCH_DEFAULT_RESULTS}, max {_SEARCH_MAX_RESULTS})."
                    ),
                },
            },
            "required": ["query"],
        },
    },
})

FUNCTIONS = {"fetch_url": fetch_url, "web_search": web_search}
