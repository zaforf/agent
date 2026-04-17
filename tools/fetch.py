"""Web fetch tool — retrieve a URL and return readable plain text.

Default behaviour: the full page is passed to a fast long-context summarizer
(gemini-2.0-flash, 1 M-token context) together with a caller-supplied prompt,
so the model receives exactly the information it asked for rather than a
truncated chunk of raw HTML text.

Pass raw=True to get the unprocessed text with offset-based pagination instead.
"""
import logging
import re
import time
from html.parser import HTMLParser

_THINK_RE = re.compile(
    r"<(thought|think|thinking)[\s>].*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

import httpx

log = logging.getLogger(__name__)

_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "article"}

# Summarizer: gemma-4-26b — MoE (3.8B active params), fast, large context
_SUMMARIZER_MODEL      = "gemma-4-26b-a4b-it"
_SUMMARIZER_CHAR_LIMIT = 128_000   # chars fed to summarizer (well within model limit)
_RAW_CHAR_LIMIT        = 8_000     # chars returned in raw/paginated mode
_FETCH_RETRIES         = 4
_FETCH_TIMEOUT_S       = 15


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


def _summarize_content(text: str, prompt: str) -> str:
    """Send page text to the Gemini native API with the caller's extraction prompt."""
    from config import GEMINI_API_KEY

    if len(text) > _SUMMARIZER_CHAR_LIMIT:
        text = (
            text[:_SUMMARIZER_CHAR_LIMIT]
            + f"\n[…content truncated at {_SUMMARIZER_CHAR_LIMIT:,} chars]"
        )

    # Use native Gemini endpoint with ?key= — new AI Studio keys (AQ. prefix)
    # don't work with Bearer auth on the OpenAI compat endpoint.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{_SUMMARIZER_MODEL}:generateContent"
    payload = {
        "systemInstruction": {
            "parts": [{"text": (
                "You are a precise document analyst. "
                "Extract and summarize exactly what the user requests from the provided content. "
                "Be comprehensive, accurate, and well-structured."
            )}],
        },
        "contents": [
            {"role": "user", "parts": [{"text": f"{prompt}\n\n---\n\n{text}"}]},
        ],
        "generationConfig": {"maxOutputTokens": 8192},
    }

    log.info("fetch-summarize  %.80s", prompt)
    with httpx.Client(timeout=60) as client:
        for attempt in range(3):
            resp = client.post(url, params={"key": GEMINI_API_KEY}, json=payload)
            if resp.status_code == 429 and attempt < 2:
                wait = 2 ** attempt
                log.warning("fetch summarizer: 429 rate limit, retrying in %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            candidates = resp.json().get("candidates", [])
            if not candidates:
                raise RuntimeError("no candidates in summarizer response")
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
            return content.strip()

    raise RuntimeError("fetch summarizer: all retries exhausted")


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

    With raw=True or no prompt: returns up to 4,000 chars starting from offset.
    Use offset pagination when you need the raw text in chunks.
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

FUNCTIONS = {"fetch_url": fetch_url}
