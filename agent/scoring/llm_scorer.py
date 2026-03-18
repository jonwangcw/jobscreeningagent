"""LLM backends and score parsing. Backend selected via config.llm.provider."""
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agent.db.repository import ScoreResult

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, prefill: str = "") -> str:
        """Send a system+user message pair; return the assistant response string.

        prefill: if non-empty, inject an assistant turn containing this string
        before the model generates. The model continues from that point.
        The returned string includes the prefill prepended so callers always
        receive a complete, parseable response.

        Only ClaudeBackend honours prefill natively. Other backends accept the
        parameter for interface compatibility but ignore it.
        """
        ...


class ClaudeBackend(LLMBackend):
    def __init__(self, model: str, max_tokens: int) -> None:
        import anthropic  # type: ignore

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str, prefill: str = "") -> str:
        messages: list[dict] = [{"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
        )
        text = msg.content[0].text
        # Claude's response does not include the prefill itself, so we prepend it
        return prefill + text if prefill else text


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str, max_tokens: int) -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str, prefill: str = "") -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class OllamaBackend(LLMBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        import httpx  # type: ignore

        self._client = httpx.Client(base_url=base_url, timeout=120)
        self._model = model

    def complete(self, system: str, user: str, prefill: str = "") -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def build_llm_backend(config: dict[str, Any]) -> LLMBackend:
    """Factory — reads provider from config dict."""
    provider = config.get("provider", "claude")
    model = config.get("model", "claude-sonnet-4-20250514")
    max_tokens = config.get("max_tokens", 2048)

    if provider == "claude":
        return ClaudeBackend(model=model, max_tokens=max_tokens)
    elif provider == "openai":
        return OpenAIBackend(model=model, max_tokens=max_tokens)
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaBackend(model=model, base_url=base_url)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")


def _strip_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_score_response(raw: str) -> ScoreResult:
    """Parse the LLM JSON response into a ScoreResult. Returns null scores on failure."""
    try:
        data = json.loads(_strip_fences(raw))
        return ScoreResult(
            role_score=float(data["role_score"]),
            location_score=float(data["location_score"]),
            stack_score=float(data["stack_score"]),
            composite_score=float(data["composite_score"]),
            rationale=str(data.get("rationale", "")),
            skill_gaps=list(data.get("skill_gaps", [])),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.error("Failed to parse LLM score response: %s | raw=%r", exc, raw[:500])
        return ScoreResult(
            role_score=None,
            location_score=None,
            stack_score=None,
            composite_score=None,
            rationale=f"parse_error: {exc}",
            skill_gaps=[],
        )
