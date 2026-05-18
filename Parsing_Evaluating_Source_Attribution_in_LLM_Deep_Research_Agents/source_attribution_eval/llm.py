"""Gemini wrapper (new `google-genai` SDK) with grounding + inline-citation injection.

The deprecated `google-generativeai` package does not support the
`google_search` grounding tool used by Gemini 2.5 series. We use the
new `google-genai` SDK instead.

Key methods:
  - generate_grounded_markdown(model, prompt, system) — Phase 0:
      Returns markdown text that already has `[text](url)` inline citations
      synthesized from `grounding_metadata.grounding_supports`. Gemini does
      NOT emit inline markdown citations on its own when grounding; we
      post-process the grounding metadata into them so the AST parser sees
      what it expects.
  - judge_json(prompt, system) — Phase 2 LLM-as-a-judge, JSON output, no tools.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential


_CLIENT: genai.Client | None = None


def _client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it."
            )
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


@dataclass
class LLMConfig:
    research_temperature: float = 0.4
    research_max_output_tokens: int = 8192
    judge_model: str = "gemini-2.5-pro"
    judge_temperature: float = 0.0
    request_timeout_seconds: int = 180


class GeminiClient:
    """Thin Gemini wrapper using the new `google-genai` SDK."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    # ----------------------------------------------------- Phase 0 (research)
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=10))
    def generate_grounded_markdown(self, model_name: str, prompt: str,
                                   system: str | None = None
                                   ) -> tuple[str, list[str]]:
        """Generate a research report with google_search grounding.

        Returns (markdown_with_inline_citations, list_of_source_urls).
        Hard request timeout via http_options to avoid indefinite hangs.
        """
        http_options = types.HttpOptions(
            timeout=self.cfg.request_timeout_seconds * 1000  # ms
        )
        cfg = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=self.cfg.research_temperature,
            max_output_tokens=self.cfg.research_max_output_tokens,
            system_instruction=system,
            http_options=http_options,
        )
        resp = _client().models.generate_content(
            model=model_name, contents=prompt, config=cfg,
        )
        raw_text = _safe_text(resp)
        gm = _get_grounding_metadata(resp)
        markdown = _inject_inline_citations(raw_text, gm)
        urls = _collect_chunk_urls(gm)
        return markdown, urls

    # ----------------------------------------------------- Phase 2 (judge)
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=10))
    def judge_json(self, prompt: str, *, system: str | None = None) -> Any:
        http_options = types.HttpOptions(timeout=60 * 1000)  # 60s hard cap per judge
        cfg = types.GenerateContentConfig(
            temperature=self.cfg.judge_temperature,
            response_mime_type="application/json",
            system_instruction=system,
            http_options=http_options,
        )
        resp = _client().models.generate_content(
            model=self.cfg.judge_model, contents=prompt, config=cfg,
        )
        raw = _safe_text(resp)
        return _extract_json(raw)


# ---------------------------------------------------------------------- helpers

def _safe_text(resp: Any) -> str:
    """Best-effort text extraction; google-genai responses have a `.text` property."""
    text = getattr(resp, "text", None)
    if text:
        return text
    try:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts)
    except Exception as exc:                                # pragma: no cover
        raise RuntimeError(f"Gemini returned no text. {resp}") from exc


def _get_grounding_metadata(resp: Any) -> Any:
    try:
        return resp.candidates[0].grounding_metadata
    except (AttributeError, IndexError):
        return None


def _collect_chunk_urls(gm: Any) -> list[str]:
    if gm is None:
        return []
    chunks = getattr(gm, "grounding_chunks", None) or []
    urls: list[str] = []
    seen: set[str] = set()
    for ch in chunks:
        w = getattr(ch, "web", None)
        if not w:
            continue
        u = getattr(w, "uri", None)
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    return urls


def _inject_inline_citations(text: str, gm: Any) -> str:
    """Use grounding_supports to add `[label](url)` after each grounded span.

    Gemini's grounding metadata exposes:
      grounding_chunks  : list of {web: {uri, title}}
      grounding_supports: list of {segment: {start_index, end_index, text},
                                   grounding_chunk_indices: [int]}

    Segment offsets are **byte-level** offsets into the response text (UTF-8).
    We splice citations in from end to start to keep offsets valid.

    Multiple supporting chunks per segment are collapsed into a single
    "[source](url) [source](url)" run, max 3 to keep readability sane.
    """
    if gm is None or not text:
        return text
    chunks = getattr(gm, "grounding_chunks", None) or []
    supports = getattr(gm, "grounding_supports", None) or []
    if not chunks or not supports:
        return text

    chunk_urls: list[str] = []
    for ch in chunks:
        w = getattr(ch, "web", None)
        chunk_urls.append((w.uri or "") if w else "")

    # Build insertions ordered by descending byte offset to splice safely
    text_bytes = bytearray(text.encode("utf-8"))
    insertions: list[tuple[int, bytes]] = []
    for sup in supports:
        seg = getattr(sup, "segment", None)
        if seg is None:
            continue
        end_idx = getattr(seg, "end_index", None)
        if end_idx is None or end_idx > len(text_bytes):
            continue
        chunk_idxs = getattr(sup, "grounding_chunk_indices", None) or []
        cites: list[str] = []
        for ci in chunk_idxs[:3]:
            if 0 <= ci < len(chunk_urls) and chunk_urls[ci]:
                cites.append(f"[source]({chunk_urls[ci]})")
        if not cites:
            continue
        insertions.append((end_idx, (" " + " ".join(cites)).encode("utf-8")))

    # Splice from the back so byte indices stay valid
    for end_idx, blob in sorted(insertions, key=lambda x: x[0], reverse=True):
        text_bytes[end_idx:end_idx] = blob

    return text_bytes.decode("utf-8", errors="replace")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")
