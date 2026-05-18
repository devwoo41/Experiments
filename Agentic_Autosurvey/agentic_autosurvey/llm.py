"""Thin Gemini wrapper with hybrid Pro/Flash routing and retries.

Uses google-generativeai. Two named tiers:
  - "writer" (default Pro, heavy reasoning, long context)
  - "light"  (default Flash, search/cluster/evaluation)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


_INIT_DONE = False


def _ensure_init() -> None:
    global _INIT_DONE
    if _INIT_DONE:
        return
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or `export GEMINI_API_KEY=...`."
        )
    genai.configure(api_key=api_key)
    _INIT_DONE = True


@dataclass
class LLMConfig:
    writer_model: str = "gemini-2.5-pro"
    light_model: str = "gemini-2.5-flash"
    temperature_writer: float = 0.4
    temperature_light: float = 0.2

    @classmethod
    def from_yaml(cls, models_cfg: dict[str, Any]) -> "LLMConfig":
        return cls(
            writer_model=os.environ.get("GEMINI_PRO_MODEL", models_cfg.get("writer", cls.writer_model)),
            light_model=os.environ.get("GEMINI_FLASH_MODEL", models_cfg.get("light", cls.light_model)),
            temperature_writer=float(models_cfg.get("temperature_writer", cls.temperature_writer)),
            temperature_light=float(models_cfg.get("temperature_light", cls.temperature_light)),
        )


class GeminiLLM:
    """Hybrid Gemini client.

    Call .generate(prompt, tier=...) where tier is "writer" or "light".
    Use .generate_json(...) to coerce a JSON object out of the model.
    """

    def __init__(self, cfg: LLMConfig):
        _ensure_init()
        self.cfg = cfg
        self._writer = genai.GenerativeModel(cfg.writer_model)
        self._light = genai.GenerativeModel(cfg.light_model)

    def _model(self, tier: str):
        return self._writer if tier == "writer" else self._light

    def _temperature(self, tier: str) -> float:
        return self.cfg.temperature_writer if tier == "writer" else self.cfg.temperature_light

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def generate(self, prompt: str, *, tier: str = "light", system: str | None = None,
                 max_output_tokens: int | None = None) -> str:
        model = self._model(tier)
        contents = prompt if not system else f"{system}\n\n---\n\n{prompt}"
        gen_config: dict[str, Any] = {"temperature": self._temperature(tier)}
        if max_output_tokens:
            gen_config["max_output_tokens"] = max_output_tokens
        resp = model.generate_content(contents, generation_config=gen_config)
        text = getattr(resp, "text", None)
        if not text:
            # Fallback for blocked or empty responses
            try:
                text = "".join(part.text for part in resp.candidates[0].content.parts)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Gemini returned no text: {resp}") from exc
        return text

    def generate_json(self, prompt: str, *, tier: str = "light", system: str | None = None) -> Any:
        """Generate then parse JSON, tolerating ```json fences and trailing text."""
        raw = self.generate(prompt, tier=tier, system=system)
        return _extract_json(raw)

    # --- embeddings ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def embed(self, texts: list[str], model: str = "gemini-embedding-001") -> list[list[float]]:
        out: list[list[float]] = []
        # The embed_content API accepts a single string; batch sequentially
        # to stay within free-tier rate limits.
        for t in texts:
            r = genai.embed_content(model=model, content=t, task_type="clustering")
            out.append(r["embedding"])
            time.sleep(0.05)
        return out


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # ```json ... ``` fence
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")
