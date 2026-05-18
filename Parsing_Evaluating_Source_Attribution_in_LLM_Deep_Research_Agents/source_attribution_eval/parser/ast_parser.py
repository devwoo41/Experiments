"""Phase 1 — Markdown AST parser (paper §3.2, Algorithm 1).

Pipeline (paper-verbatim):
  1. Canonicalize — normalize whitespace, strip code blocks.
  2. Build AST (markdown-it-py token stream).
  3. Extract citation nodes across 5 formats:
        - numbered references     [1], [2]
        - footnote-style          [^note]
        - inline markdown link    [text](url)
        - autolinks               <https://...>
        - ranges                  [1-3]  (expanded to [1], [2], [3])
  4. Sentence-level segmentation.
  5. Backward attribution — a citation at end of passage covers all
     preceding uncited sentences in that passage.
  6. Deduplicated citation registry with normalized URLs.

No LLM is used in this phase (paper: "structurally extracts citation-claim
pairs without requiring LLM inference").
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from markdown_it import MarkdownIt

from ..state import Attribution, AttributionDocument, Citation


# Regexes for the five citation formats in the paper
RE_INLINE_LINK = re.compile(r"\[([^\[\]]+)\]\((https?://[^\s)]+)\)")
RE_AUTOLINK    = re.compile(r"<(https?://[^>\s]+)>")
RE_NUM_RANGE   = re.compile(r"\[(\d+)\s*[\-–]\s*(\d+)\]")
RE_NUM_REF     = re.compile(r"\[(\d+)\]")
RE_FOOTNOTE    = re.compile(r"\[\^([A-Za-z0-9_\-]+)\]")

# Reference list lines such as `[1]: https://example.com` or `[1] https://...`
RE_REF_LIST    = re.compile(
    r"^\s*\[(\d+|\^[A-Za-z0-9_\-]+)\]\s*[:.\)\-]?\s*(https?://\S+)", re.MULTILINE
)


def parse_markdown_report(md_text: str, query_id: str, query: str,
                          model: str, depth: str, *,
                          backward_attribution: bool = True,
                          segmenter: str = "regex") -> AttributionDocument:
    """Run the full Phase-1 pipeline and return an AttributionDocument."""

    # ----- 1. Canonicalize: normalize line endings, strip code blocks -----
    md_canon = _canonicalize(md_text)

    # ----- 2/3. Build AST and extract citation labels -----
    # Implementation note: we operate on the canonicalized text plus
    # markdown-it-py token stream to (a) confirm the document parses and
    # (b) keep code-fence stripping consistent with what the AST sees.
    md_parser = MarkdownIt("commonmark", {"breaks": False, "html": False})
    tokens = md_parser.parse(md_canon)
    # We don't need tokens further — canonicalize already removed code blocks
    # and we rely on regex for citation extraction across formats.
    _ = tokens

    # 3a. Build a numbered/footnote reference registry from the reference list
    ref_to_url: dict[str, str] = {}
    for m in RE_REF_LIST.finditer(md_canon):
        label, url = m.group(1), _normalize_url(m.group(2))
        ref_to_url[label] = url

    # Strip the reference list from the body so it's not segmented as claims
    body_text = RE_REF_LIST.sub("", md_canon).strip()

    # 3b. Expand [1-3] ranges into [1][2][3] before further extraction
    body_text = RE_NUM_RANGE.sub(_expand_range, body_text)

    # ----- 4/5. Sentence segmentation + backward attribution -----
    citations: dict[str, Citation] = {}  # url -> Citation
    raw_label_index: dict[str, str] = {} # raw label string -> url (for dedup)

    paragraphs = _split_paragraphs(body_text)

    attributions: list[Attribution] = []
    next_aid = 1
    next_cid = 1

    def _register_citation(url: str, raw_label: str) -> str:
        nonlocal next_cid
        url = _normalize_url(url)
        if url in citations:
            cit = citations[url]
            if raw_label and raw_label not in cit["raw_labels"]:
                cit["raw_labels"].append(raw_label)
            return cit["citation_id"]
        cid = f"c{next_cid}"
        next_cid += 1
        citations[url] = Citation(citation_id=cid, raw_labels=[raw_label] if raw_label else [],
                                   url=url, url_content=None)
        return cid

    for para in paragraphs:
        sentences = _segment(para, backend=segmenter)
        sent_records: list[dict[str, Any]] = []
        # For each sentence collect: text, text_nocite, [citation_ids_directly_in_sentence]
        for s_text in sentences:
            cids: list[str] = []
            # inline markdown link
            for m in RE_INLINE_LINK.finditer(s_text):
                cids.append(_register_citation(m.group(2), m.group(0)))
            # autolink
            for m in RE_AUTOLINK.finditer(s_text):
                cids.append(_register_citation(m.group(1), m.group(0)))
            # numbered ref -> lookup in registry
            for m in RE_NUM_REF.finditer(s_text):
                label = m.group(1)
                url = ref_to_url.get(label)
                if url:
                    cids.append(_register_citation(url, f"[{label}]"))
            # footnote ref
            for m in RE_FOOTNOTE.finditer(s_text):
                label = m.group(1)
                url = ref_to_url.get(f"^{label}") or ref_to_url.get(label)
                if url:
                    cids.append(_register_citation(url, f"[^{label}]"))
            # Deduplicate within the sentence preserving order
            seen, deduped = set(), []
            for c in cids:
                if c not in seen:
                    seen.add(c); deduped.append(c)
            sent_records.append({
                "text": s_text,
                "text_nocite": _strip_citations(s_text),
                "citation_ids": deduped,
            })

        # ----- Backward attribution within this paragraph (paper §3.2) -----
        if backward_attribution:
            # Find the LAST sentence that has citations; propagate its citations
            # back to every preceding *uncited* sentence within this paragraph.
            # We propagate only when there is at least one preceding uncited
            # sentence and the last sentence with citations marks the trailing
            # cluster.
            last_cited_idx = -1
            for i, r in enumerate(sent_records):
                if r["citation_ids"]:
                    last_cited_idx = i
            if last_cited_idx >= 0:
                trailing_cits = sent_records[last_cited_idx]["citation_ids"]
                for j in range(last_cited_idx):
                    if not sent_records[j]["citation_ids"]:
                        sent_records[j]["citation_ids"] = list(trailing_cits)

        for r in sent_records:
            if not r["citation_ids"]:
                continue  # uncited sentences are not attributions to evaluate
            attributions.append(Attribution(
                attribution_id=f"a{next_aid}",
                text=r["text"],
                text_nocite=r["text_nocite"],
                citation_ids=r["citation_ids"],
            ))
            next_aid += 1

    doc: AttributionDocument = {
        "query_id": query_id,
        "query": query,
        "model": model,
        "depth": depth,
        "raw_markdown": md_text,
        "citations": list(citations.values()),
        "attributions": attributions,
        "evals": [],
    }
    return doc


# ---------------------------------------------------------------- canonicalize

_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INDENTED = re.compile(r"(?:^|\n)((?:    [^\n]*\n?)+)")


def _canonicalize(md: str) -> str:
    """Normalize line endings, strip code blocks (paper §3.2)."""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = _FENCED.sub("", md)
    md = _INDENTED.sub("\n", md)
    # Collapse runs of more than two blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ---------------------------------------------------------------- URL utils

def _normalize_url(u: str) -> str:
    try:
        p = urlparse(u.strip())
        # Strip trailing punctuation that often clings to URLs in prose
        path = p.path.rstrip(").,;:")
        return urlunparse((p.scheme, p.netloc.lower(), path, p.params, p.query, ""))
    except Exception:
        return u.strip()


def _expand_range(m: re.Match[str]) -> str:
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi or hi - lo > 50:  # sanity cap
        return m.group(0)
    return "".join(f"[{i}]" for i in range(lo, hi + 1))


# ---------------------------------------------------------------- paragraphs

def _split_paragraphs(text: str) -> list[str]:
    """Split body into passages on blank lines. Heading lines are their own passage."""
    out: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Drop pure heading lines from claim segmentation (they aren't claims)
        if re.match(r"^#{1,6}\s", chunk) and "\n" not in chunk:
            continue
        # If chunk starts with a heading + content, strip leading heading line
        chunk = re.sub(r"^#{1,6}\s[^\n]*\n+", "", chunk)
        if chunk:
            out.append(chunk)
    return out


# ---------------------------------------------------------------- segmentation

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def _segment(text: str, *, backend: str = "regex") -> list[str]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    if backend == "nltk":
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except Exception:
            pass  # fall through to regex
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# ---------------------------------------------------------------- citation strip

_STRIP_PATTERNS = [
    RE_INLINE_LINK,    # [text](url) -> text
    RE_AUTOLINK,       # <url> -> ""
    RE_NUM_REF,        # [1] -> ""
    RE_FOOTNOTE,       # [^note] -> ""
]


def _strip_citations(s: str) -> str:
    out = s
    out = RE_INLINE_LINK.sub(lambda m: m.group(1), out)
    out = RE_AUTOLINK.sub("", out)
    out = RE_NUM_REF.sub("", out)
    out = RE_FOOTNOTE.sub("", out)
    return re.sub(r"\s{2,}", " ", out).strip(" ,;.")
