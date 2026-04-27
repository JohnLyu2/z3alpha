"""LLM-derived PUCT priors (stage-1): OpenAI Chat Completions over HTTP, no SDK."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You score Z3 SMT tactics for the next step. Reply with a single JSON object "
    "only: keys are exactly the tactic names given by the user, values are integers "
    "from 0 to 5 (higher = more promising). No markdown, no explanation."
)


@dataclass(frozen=True)
class LLMPriorConfig:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 30.0
    temperature: float = 0.0
    cache_path: Path | None = None


def _scores_to_priors(scores: dict[str, int]) -> dict[str, float]:
    """Map integer scores to normalized priors; all-zero -> uniform 1.0 each."""
    if not scores:
        return {}
    total = sum(scores.values())
    if total <= 0:
        return {k: 1.0 for k in scores}
    return {k: scores[k] / total for k in scores}


def _clamp_score(v: Any) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, n))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    # Strip ```json ... ``` if present
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class LLMPriorScorer:
    """Scores candidate tactics via OpenAI REST API; caches on disk + memory."""

    def __init__(self, cfg: LLMPriorConfig) -> None:
        self._cfg = cfg
        self._memory: dict[str, dict[str, float]] = {}
        self._api_key = os.environ.get(cfg.api_key_env, "")
        if cfg.enabled and not self._api_key:
            log.warning(
                "LLM prior enabled but %s is unset; using uniform priors until set.",
                cfg.api_key_env,
            )
        if cfg.cache_path is not None:
            self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        path = self._cfg.cache_path
        assert path is not None
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Could not load LLM prior cache %s: %s", path, e)
            return
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, dict):
                    self._memory[k] = {str(a): float(p) for a, p in v.items()}

    def _save_disk_cache(self) -> None:
        path = self._cfg.cache_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._memory, f, indent=0, sort_keys=True)

    def _cache_key(
        self, logic: str, partial_strategy: str, candidate_actions: list[str]
    ) -> str:
        payload = [
            logic,
            partial_strategy,
            sorted(candidate_actions),
            self._cfg.model,
        ]
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call_chat_completions(self, user_content: str) -> str | None:
        if not self._api_key:
            return None
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self._cfg.temperature,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            log.warning("OpenAI chat request failed: %s", e)
            return None
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("OpenAI response was not JSON")
            return None
        choices = outer.get("choices")
        if not choices or not isinstance(choices, list):
            return None
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        return content if isinstance(content, str) else None

    def _uniform(self, candidate_actions: list[str]) -> dict[str, float]:
        return {a: 1.0 for a in candidate_actions}

    def score(
        self,
        logic: str,
        partial_strategy: str,
        candidate_actions: list[str],
    ) -> dict[str, float]:
        """Return normalized prior per candidate name; uniform on any failure."""
        if not candidate_actions:
            return {}
        key = self._cache_key(logic, partial_strategy, candidate_actions)
        if key in self._memory:
            cached = self._memory[key]
            if all(a in cached for a in candidate_actions):
                return {a: cached[a] for a in candidate_actions}

        if not self._api_key:
            u = self._uniform(candidate_actions)
            self._memory[key] = u
            self._save_disk_cache()
            return u

        user_lines = [
            f"Logic: {logic}",
            "",
            "Current partial Z3 strategy (SMT-LIB tactic script):",
            partial_strategy,
            "",
            "Candidate tactic names for the next step (JSON keys must match exactly):",
            json.dumps(candidate_actions),
        ]
        user_content = "\n".join(user_lines)
        content = self._call_chat_completions(user_content)
        if content is None:
            u = self._uniform(candidate_actions)
            self._memory[key] = u
            self._save_disk_cache()
            return u

        parsed = _extract_json_object(content)
        if parsed is None:
            log.warning("LLM prior: could not parse JSON from model output")
            u = self._uniform(candidate_actions)
            self._memory[key] = u
            self._save_disk_cache()
            return u

        scores: dict[str, int] = {}
        for a in candidate_actions:
            if a in parsed:
                scores[a] = _clamp_score(parsed[a])
            else:
                scores[a] = 0

        priors = _scores_to_priors(scores)
        self._memory[key] = priors
        self._save_disk_cache()
        return {a: priors[a] for a in candidate_actions}
