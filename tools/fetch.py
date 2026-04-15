"""Web fetch tool — retrieve a URL and return readable plain text."""
import re
from html.parser import HTMLParser

import httpx

_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "article"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._depth = 0   # nesting depth of skip tags

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


def fetch_url(url: str) -> str:
    """Fetch a URL and return its readable text content (max ~8 000 chars)."""
    MAX = 8_000
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentFetch/1.0)"}
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=12)
        resp.raise_for_status()

        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            parser = _TextExtractor()
            parser.feed(resp.text)
            text = parser.get_text()
        else:
            text = resp.text.strip()

        if len(text) > MAX:
            text = text[:MAX] + f"\n\n[… {len(text) - MAX} more characters truncated]"

        return text or "(no readable content)"

    except httpx.TimeoutException:
        return f"Error: request to {url} timed out (12 s)"
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
                "Fetch the text content of a URL. Use when the user shares a link, "
                "asks about a specific page, or wants you to read an article or document. "
                "Returns readable plain text (up to ~8 000 characters)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch"},
                },
                "required": ["url"],
            },
        },
    }
]

FUNCTIONS = {"fetch_url": fetch_url}
