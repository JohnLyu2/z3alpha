"""LLM-derived PUCT priors via OpenAI Responses API + structured outputs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

# instructions = system; input = user (Responses API)
_INSTRUCTIONS = (
    "You score Z3 SMT tactics for the next MCTS search step. "
    "Output must follow the response schema: a list of scores, with exactly one entry "
    "per candidate tactic name from the user message. "
    "Each value is an integer from 0 to 5 (higher = more promising). "
    "tactic_name strings must match the provided candidate names exactly (character-for-character)."
)


class TacticScoreItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tactic_name: str = Field(
        description="Z3 tactic name from the candidate list, match exactly (e.g. smt, ctx-simplify)"
    )
    value: int = Field(ge=0, le=5, description="Score 0-5, higher = more promising")


class TacticPriorScores(BaseModel):
    """Structured output: see OpenAI structured outputs (JSON Schema via Pydantic)."""

    model_config = ConfigDict(extra="forbid")

    scores: list[TacticScoreItem] = Field(
        description="One object per candidate tactic, tactic_name + value 0-5"
    )


@dataclass(frozen=True)
class LLMPriorConfig:
    enabled: bool = False
    model: str = "gpt-5.4-mini"
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


class LLMPriorScorer:
    """Scores candidate tactics via OpenAI Responses API; caches on disk + memory."""

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

    def _responses_parse_scores(
        self, user_content: str
    ) -> TacticPriorScores | None:
        """Call ``POST /v1/responses`` with ``text_format`` = Pydantic schema (structured outputs)."""
        if not self._api_key:
            return None
        client = OpenAI(api_key=self._api_key, base_url=self._cfg.base_url.rstrip("/"))
        try:
            response = client.responses.parse(
                model=self._cfg.model,
                instructions=_INSTRUCTIONS,
                input=user_content,
                text_format=TacticPriorScores,
                temperature=self._cfg.temperature,
                timeout=self._cfg.timeout_s,
            )
        except Exception as e:
            log.warning("OpenAI Responses request failed: %s", e)
            return None

        status = getattr(response, "status", None)
        if status and status != "completed":
            log.warning("OpenAI Responses status=%s (expected completed); using uniform", status)
            return None
        if getattr(response, "error", None) is not None:
            log.warning("OpenAI Responses error field set; using uniform")
            return None
        inc = getattr(response, "incomplete_details", None)
        if inc is not None:
            log.warning("OpenAI Responses incomplete: %s; using uniform", inc)
            return None

        parsed: TacticPriorScores | None = response.output_parsed
        if parsed is None:
            log.warning("LLM prior: no structured output (refusal, parse miss, or empty)")
            return None
        return parsed

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

        user_content = "\n".join(
            [
                f"Logic: {logic}",
                "",
                "Current partial Z3 strategy (SMT-LIB tactic script):",
                partial_strategy,
                "",
                "Candidate Z3 tactic names for the next step. Include each name exactly once in "
                "`scores` with a 0-5 value (names are JSON strings, e.g. \"smt\", \"ctx-simplify\"):",
                json.dumps(candidate_actions),
            ]
        )
        parsed = self._responses_parse_scores(user_content)
        if parsed is None:
            u = self._uniform(candidate_actions)
            self._memory[key] = u
            self._save_disk_cache()
            return u

        by_name: dict[str, int] = {}
        for item in parsed.scores:
            by_name[str(item.tactic_name)] = _clamp_score(item.value)

        scores: dict[str, int] = {}
        for a in candidate_actions:
            scores[a] = by_name.get(a, 0)

        priors = _scores_to_priors(scores)
        self._memory[key] = priors
        self._save_disk_cache()
        return {a: priors[a] for a in candidate_actions}
