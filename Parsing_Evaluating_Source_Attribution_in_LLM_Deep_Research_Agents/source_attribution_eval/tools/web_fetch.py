"""Web content extraction (paper §3.3.1 + §3.3.2).

We use Trafilatura — a high-quality HTML→text extractor — as our "web
content extractor". Returns `(status_code, plain_text)`. JS-rendered SPAs
return whatever the static HTML contains; the paper notes JS rendering as
an environment requirement but Trafilatura's quality on the long tail of
documentation/blogs/PDFs is generally sufficient.
"""

from __future__ import annotations

import requests
import trafilatura


def http_probe(url: str, *, timeout: int, user_agent: str) -> tuple[int, str]:
    """Lightweight GET probe. Returns (status_code, body_or_empty).

    Hard timeout: (connect=5, read=timeout). Skips HEAD because many servers
    return spurious 403/405 for HEAD even when GET works, and HEAD doubled
    our chance of hanging. No tenacity retries: a stalled URL is a stalled
    URL — we bound failure fast so the ThreadPool doesn't deadlock the run.
    """
    headers = {"User-Agent": user_agent}
    connect_to = 5
    read_to = max(5, int(timeout))
    try:
        # stream=True so we can cap body size without downloading huge files
        g = requests.get(
            url, headers=headers, timeout=(connect_to, read_to),
            allow_redirects=True, stream=True,
        )
        # Cap body at 2 MB to keep parser fast on long pages
        chunks: list[bytes] = []
        total = 0
        for chunk in g.iter_content(chunk_size=64 * 1024):
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= 2 * 1024 * 1024:
                break
        try:
            body = b"".join(chunks).decode(g.encoding or "utf-8", errors="replace")
        except Exception:
            body = ""
        return g.status_code, body
    except requests.RequestException:
        return 0, ""
    except Exception:
        return 0, ""


def extract_main_text(html: str, *, url: str | None = None) -> str:
    """Trafilatura extraction. Returns plain text or empty string on failure."""
    if not html:
        return ""
    extracted = trafilatura.extract(
        html,
        url=url,
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_recall=True,
        no_fallback=False,
    )
    return extracted or ""
