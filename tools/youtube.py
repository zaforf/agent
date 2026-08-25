"""YouTube transcript tool — fetch metadata and plain-text transcript via Supadata API.

Default (with prompt): passes the metadata and full transcript to the Gemma summarizer so
the model receives grounded context and exactly what it asked for rather than raw text.
Raw mode: returns metadata and up to _RAW_CHAR_LIMIT chars from offset, with the same
pagination note pattern as fetch_url.

Accepts a bare video ID (e.g. dQw4w9WgXcQ) or any YouTube URL; strips
the ID and constructs a canonical URL for the Supadata client.
"""
import logging
import re

from supadata import Supadata
from supadata.errors import SupadataError
from supadata.types import BatchJob

import config
from summarizer import summarize_gemma

log = logging.getLogger(__name__)

_RAW_CHAR_LIMIT        = 8_000
_SUMMARIZER_CHAR_LIMIT = 256_000

# Matches the 11-char video ID in any common YouTube URL form.
_YT_URL_RE = re.compile(
    r'(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|v/)|youtu\.be/)'
    r'([A-Za-z0-9_-]{11})'
)

_SUMMARIZER_SYSTEM = (
    "You are a precise transcript analyst. "
    "Extract and summarize exactly what the user requests from the provided transcript. "
    "Be comprehensive, accurate, and well-structured."
)


def _to_url(video_id_or_url: str) -> str:
    """Return a canonical youtube.com watch URL for a bare ID or any YT URL."""
    s = (video_id_or_url or "").strip()
    m = _YT_URL_RE.search(s)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    # Bare 11-char ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return f"https://www.youtube.com/watch?v={s}"
    # Pass through unknown forms and let the API surface the error.
    return s


def youtube_transcript(
    video_id: str, prompt: str = "", offset: int = 0, raw: bool = False
) -> str:
    """Fetch the metadata and plain-text transcript of a YouTube video.

    With a prompt (default): passes metadata and the full transcript to a summarizer that
    extracts exactly what was asked. Preferred for focused extraction.

    With raw=True or no prompt: returns metadata and up to _RAW_CHAR_LIMIT chars from
    offset. Use offset pagination when you need the raw text in chunks.
    """
    if not config.SUPADATA_API_KEY:
        return "Error: youtube_transcript disabled — set SUPADATA_API_KEY in .env"

    url = _to_url(video_id)
    try:
        client = Supadata(api_key=config.SUPADATA_API_KEY)
        
        # Fetch metadata for grounding
        meta_res = client.metadata(url=url)
        author = meta_res.author
        author_name = (author.display_name or author.username or "Unknown") if author else "Unknown"
        meta_header = (
            f"TITLE: {meta_res.title or 'Unknown'}\n"
            f"CHANNEL: {author_name}\n"
            f"DESCRIPTION: {meta_res.description or 'No description available'}\n"
            f"---"
        )

        # Fetch transcript
        result = client.transcript(url=url, text=True)
        if isinstance(result, BatchJob):
            return f"{meta_header}\n(transcript is being generated — try again in a few moments)"
        text = result.content if isinstance(result.content, str) else ""
    except SupadataError as e:
        return f"Error: {e.message}"
    except Exception as e:
        return f"Error fetching video data for {url}: {e}"

    if not text:
        # Still return metadata even if transcript is missing
        return f"{meta_header}\n(no transcript available)" if meta_header else "(no transcript available)"

    degraded_note = ""
    if prompt and not raw:
        if len(text) > _SUMMARIZER_CHAR_LIMIT:
            text = (
                text[:_SUMMARIZER_CHAR_LIMIT]
                + f"\n[…transcript truncated at {_SUMMARIZER_CHAR_LIMIT:,} chars]"
            )
        try:
            log.info("youtube-transcript-summarize  %.80s", prompt)
            # Include metadata in the summarization prompt for better grounding
            return summarize_gemma(_SUMMARIZER_SYSTEM, f"{meta_header}\n\n{prompt}\n\n---\n\n{text}")
        except Exception as e:
            log.warning("youtube_transcript: summarizer failed (%s), falling back to raw", e)
            degraded_note = (
                "[youtube_transcript note: structured transcript summarization was unavailable; "
                "this is raw paginated transcript text and may be incomplete.]\n\n"
            )

    total = len(text)
    chunk = text[offset:]
    if len(chunk) > _RAW_CHAR_LIMIT:
        remaining   = total - offset - _RAW_CHAR_LIMIT
        next_offset = offset + _RAW_CHAR_LIMIT
        chunk = (
            chunk[:_RAW_CHAR_LIMIT]
            + f"\n\n[… {remaining:,} more chars — call youtube_transcript with offset={next_offset} to continue]"
        )
    
    # Prepend metadata to raw results
    return f"{meta_header}\n\n{degraded_note}{chunk}"


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "youtube_transcript",
            "description": (
                "Fetch metadata (title, channel, description) and the plain-text transcript of a YouTube video. "
                "Always provide a prompt describing what to extract or summarize. "
                "Accepts a bare video ID (e.g. dQw4w9WgXcQ) or any YouTube URL. "
                "Use raw=true only when you need unprocessed transcript text "
                "(e.g. offset-based pagination)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": (
                            "YouTube video ID (e.g. dQw4w9WgXcQ) or full YouTube URL "
                            "(youtube.com/watch?v=…, youtu.be/…, or shorts URL)."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What to extract or summarize from the transcript. "
                            "E.g. 'Summarize the key arguments' or 'List every topic discussed'. "
                            "Provide this whenever you want structured information."
                        ),
                    },
                    "raw": {
                        "type": "boolean",
                        "description": (
                            f"Return raw transcript text (up to {_RAW_CHAR_LIMIT:,} chars) "
                            "instead of summarizing. Use with offset for pagination. Default: false."
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
                "required": ["video_id"],
            },
        },
    }
]

FUNCTIONS = {"youtube_transcript": youtube_transcript}
