"""LLM-derived PUCT priors via OpenAI-compatible Chat Completions + structured outputs."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

PriorSource = Literal["api_call", "cache_hit", "api_call_failed"]

_INSTRUCTIONS = (
    "You score Z3 SMT tactics for the next MCTS search step. "
    "Output must follow the response schema: a list of scores, with exactly one entry "
    "per candidate tactic name from the user message. "
    "Each value is an integer from 0 to 10. "
    "Interpret the score as confidence that the tactic is a good next step in the current tactic chain. "
    "Use 0 or 1 for tactics that are very unlikely to help; reserve 8-10 for strong next steps. "
    "Differentiate candidates clearly—avoid giving every tactic similar mid-range scores. "
    "tactic_name strings must match the provided candidate names exactly (character-for-character)."
)


class TacticScoreItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tactic_name: str = Field(
        description="Z3 tactic name from the candidate list, match exactly (e.g. smt, ctx-simplify)"
    )
    value: int = Field(
        ge=0,
        le=10,
        description="Confidence score 0-10 that this tactic is a good next step",
    )


class TacticPriorScores(BaseModel):
    """Structured output."""

    model_config = ConfigDict(extra="forbid")

    scores: list[TacticScoreItem] = Field(
        description="One object per candidate tactic, tactic_name + value 0-10"
    )


@dataclass(frozen=True)
class LLMPriorConfig:
    enabled: bool = False
    model: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = None
    llm_timeout: float = 30.0
    temperature: float = 0.0
    softmax_temperature: float = 2.0
    uncertainty_spread_threshold: int = 1
    reject_score_below_or_equal: int = 1
    qa_log_path: str | None = None


def _default_api_key_env(base_url: str) -> str:
    if "openrouter.ai" in base_url.lower():
        return "OPENROUTER_API_KEY"
    return "OPENAI_API_KEY"


class LLMPriorScorer:
    """Scores candidate tactics via OpenAI-compatible Chat Completions API."""

    def __init__(self, cfg: LLMPriorConfig) -> None:
        self._cfg = cfg
        self._memory: dict[str, tuple[str, dict[str, float]]] = {}
        self._api_key_env = cfg.api_key_env or _default_api_key_env(cfg.base_url)
        self._api_key = os.environ.get(self._api_key_env, "")
        if cfg.enabled and not self._api_key:
            raise RuntimeError(
                f"LLM prior is enabled but {self._api_key_env!r} is not set or empty."
            )
        self._qa_log_path = Path(cfg.qa_log_path) if cfg.qa_log_path else None
        self.api_call_count = 0
        self.last_prior_source: PriorSource | None = None
        self.last_mapping_mode: str | None = None

    def _log_prior_event(
        self,
        sim_id: int | None,
        source: PriorSource,
        *,
        mapping_mode: str | None = None,
        detail: str | None = None,
    ) -> None:
        parts = [f"LLM prior sim {sim_id}: source={source}"]
        if mapping_mode is not None:
            parts.append(f"mapping={mapping_mode}")
        if detail:
            parts.append(detail)
        log.info(" ".join(parts))

    def _append_qa_log(self, payload: dict) -> None:
        if self._qa_log_path is None:
            return
        ts_utc = datetime.now(timezone.utc).isoformat()
        try:
            self._qa_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._append_qa_log_readable(ts_utc, payload)
        except Exception as e:
            log.warning("Failed writing LLM Q&A log %s: %s", self._qa_log_path, e)

    def _append_qa_log_readable(self, ts_utc: str, payload: dict) -> None:
        """Write a human-readable Q&A log for quick inspection."""
        if self._qa_log_path is None:
            return
        lines = [
            "=" * 80,
            f"time_utc: {ts_utc}",
            f"kind: {payload.get('kind', 'unknown')}",
        ]
        for key in (
            "sim_id",
            "prior_source",
            "reason",
            "mapping_mode",
            "logic",
            "partial_strategy",
            "candidate_actions",
            "scores",
            "priors",
            "ranked_priors",
            "api_key_env",
            "error",
        ):
            if key in payload:
                val = payload[key]
                if isinstance(val, (dict, list)):
                    val = json.dumps(val, ensure_ascii=False, indent=2)
                lines.append(f"{key}: {val}")
        request = payload.get("request", {})
        if isinstance(request, dict) and request.get("input"):
            lines.append("-- request.input --")
            lines.append(str(request["input"]))
        response = payload.get("response", {})
        if isinstance(response, dict):
            lines.append("-- response --")
            lines.append(json.dumps(response, ensure_ascii=False, indent=2))
        lines.append("")
        with open(self._qa_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _reject_chat_response(
        self,
        sim_id: int | None,
        reason: str,
        user_content: str,
        *,
        response: dict | None = None,
        log_detail: str | None = None,
    ) -> None:
        detail = log_detail or reason
        log.warning("Chat Completions rejected (%s); using uniform", detail)
        self._append_qa_log(
            {
                "kind": "llm_prior_response_rejected",
                "sim_id": sim_id,
                "prior_source": "api_call_failed",
                "reason": reason,
                "request": {"input": user_content},
                "response": response or {},
            }
        )

    def _chat_completions_parse_scores(
        self, user_content: str, sim_id: int | None = None
    ) -> TacticPriorScores | None:
        """Call ``POST /v1/chat/completions`` with Pydantic structured output."""
        if not self._api_key:
            self._append_qa_log(
                {
                    "kind": "llm_prior_request_skipped",
                    "sim_id": sim_id,
                    "prior_source": "api_call_failed",
                    "reason": "missing_api_key",
                    "api_key_env": self._api_key_env,
                    "request": {"input": user_content},
                }
            )
            return None
        client = OpenAI(api_key=self._api_key, base_url=self._cfg.base_url.rstrip("/"))
        messages = [
            {"role": "system", "content": _INSTRUCTIONS},
            {"role": "user", "content": user_content},
        ]
        try:
            self.api_call_count += 1
            completion = client.chat.completions.parse(
                model=self._cfg.model,
                messages=messages,
                response_format=TacticPriorScores,
                temperature=self._cfg.temperature,
                timeout=self._cfg.llm_timeout,
            )
        except (TypeError, AttributeError) as e:
            log.warning("Chat Completions malformed response: %s", e)
            self._append_qa_log(
                {
                    "kind": "llm_prior_request_error",
                    "sim_id": sim_id,
                    "prior_source": "api_call_failed",
                    "api_key_env": self._api_key_env,
                    "request": {"input": user_content},
                    "error": str(e),
                }
            )
            return None
        except Exception as e:
            log.warning("Chat Completions request failed: %s", e)
            self._append_qa_log(
                {
                    "kind": "llm_prior_request_error",
                    "sim_id": sim_id,
                    "prior_source": "api_call_failed",
                    "api_key_env": self._api_key_env,
                    "request": {"input": user_content},
                    "error": str(e),
                }
            )
            return None

        if completion is None:
            self._reject_chat_response(
                sim_id, "completion_none", user_content, log_detail="completion is None"
            )
            return None

        choices = getattr(completion, "choices", None)
        if choices is None:
            self._reject_chat_response(
                sim_id,
                "choices_none",
                user_content,
                log_detail="choices is None",
            )
            return None
        if len(choices) == 0:
            self._reject_chat_response(sim_id, "no_choices", user_content)
            return None

        choice = choices[0]
        if choice is None:
            self._reject_chat_response(sim_id, "choice_none", user_content)
            return None

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            message = getattr(choice, "message", None)
            self._reject_chat_response(
                sim_id,
                "finish_reason_length",
                user_content,
                response={
                    "finish_reason": finish_reason,
                    "content": getattr(message, "content", None) if message else None,
                },
            )
            return None

        message = getattr(choice, "message", None)
        if message is None:
            self._reject_chat_response(sim_id, "message_none", user_content)
            return None

        refusal = getattr(message, "refusal", None)
        if refusal:
            self._reject_chat_response(
                sim_id,
                "refusal",
                user_content,
                response={"refusal": refusal},
                log_detail=f"refusal={refusal!r}",
            )
            return None

        parsed: TacticPriorScores | None = getattr(message, "parsed", None)
        raw_content = getattr(message, "content", None)
        if parsed is None:
            self._reject_chat_response(
                sim_id,
                "no_structured_output",
                user_content,
                response={
                    "finish_reason": finish_reason,
                    "content": raw_content,
                },
            )
            return None

        scores_list = getattr(parsed, "scores", None)
        if scores_list is None or len(scores_list) == 0:
            self._reject_chat_response(
                sim_id,
                "empty_scores",
                user_content,
                response={
                    "finish_reason": finish_reason,
                    "content": raw_content,
                    "parsed_scores": [],
                },
            )
            return None

        self._append_qa_log(
            {
                "kind": "llm_prior_qa",
                "sim_id": sim_id,
                "prior_source": "api_call",
                "api_key_env": self._api_key_env,
                "request": {"input": user_content},
                "response": {
                    "finish_reason": finish_reason,
                    "content": raw_content,
                    "parsed_scores": [it.model_dump() for it in scores_list],
                },
            }
        )
        return parsed

    def _uniform(self, candidate_actions: list[str]) -> dict[str, float]:
        return {a: 1.0 for a in candidate_actions}

    def _ranked_priors(self, priors: dict[str, float]) -> list[dict[str, float | str]]:
        ranked = sorted(priors.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{"tactic_name": name, "prior": float(prior)} for name, prior in ranked]

    def _format_ranked_priors(self, priors: dict[str, float]) -> str:
        ranked = self._ranked_priors(priors)
        return ", ".join(f"{row['tactic_name']}={row['prior']:.3f}" for row in ranked)

    def _thresholded_softmax(
        self,
        scores: dict[str, int],
    ) -> tuple[str, dict[str, float], str | None]:
        if not scores:
            return "uniform_empty", {}, "empty_candidates"

        spread = max(scores.values()) - min(scores.values())
        if spread <= self._cfg.uncertainty_spread_threshold:
            return "uniform_uncertain", self._uniform(list(scores.keys())), "uncertain_spread"

        eligible = {
            name: score
            for name, score in scores.items()
            if score > self._cfg.reject_score_below_or_equal
        }
        if not eligible:
            return "uniform_all_rejected", self._uniform(list(scores.keys())), "all_rejected"

        tau = max(float(self._cfg.softmax_temperature), 1e-6)
        max_score = max(eligible.values())
        exp_scores = {
            name: math.exp((score - max_score) / tau) for name, score in eligible.items()
        }
        denom = sum(exp_scores.values())
        if denom <= 0:
            return "uniform_softmax_error", self._uniform(list(scores.keys())), "softmax_zero_sum"

        priors = {name: 0.0 for name in scores}
        for name, val in exp_scores.items():
            priors[name] = val / denom
        return "thresholded_softmax", priors, None

    def score(
        self,
        logic: str,
        partial_strategy: str,
        candidate_actions: list[str],
        sim_id: int | None = None,
    ) -> dict[str, float]:
        """Return prior per candidate name; uniform on failure/uncertainty."""
        if not candidate_actions:
            return {}
        if partial_strategy in self._memory:
            cached_mode, cached_priors = self._memory[partial_strategy]
            if all(a in cached_priors for a in candidate_actions):
                subset = {a: cached_priors[a] for a in candidate_actions}
                self.last_prior_source = "cache_hit"
                self.last_mapping_mode = cached_mode
                self._append_qa_log(
                    {
                        "kind": "llm_prior_cache_hit",
                        "sim_id": sim_id,
                        "prior_source": "cache_hit",
                        "mapping_mode": cached_mode,
                        "logic": logic,
                        "partial_strategy": partial_strategy,
                        "candidate_actions": candidate_actions,
                        "priors": subset,
                        "ranked_priors": self._ranked_priors(subset),
                    }
                )
                self._log_prior_event(
                    sim_id,
                    "cache_hit",
                    mapping_mode=cached_mode,
                    detail=self._format_ranked_priors(subset),
                )
                return subset

        n_candidates = len(candidate_actions)
        user_content = "\n".join(
            [
                f"Logic: {logic}",
                "",
                "Current partial Z3 strategy (SMT-LIB tactic script):",
                partial_strategy,
                "",
                f"Candidate Z3 tactic names for the next step ({n_candidates} tactics). "
                "Include each name exactly once in `scores` with a 0-10 confidence value "
                "(names are JSON strings, e.g. \"smt\", \"ctx-simplify\"):",
                json.dumps(candidate_actions),
            ]
        )
        parsed = self._chat_completions_parse_scores(user_content, sim_id=sim_id)
        if parsed is None:
            u = self._uniform(candidate_actions)
            self.last_prior_source = "api_call_failed"
            self.last_mapping_mode = "uniform_api_fallback"
            self._append_qa_log(
                {
                    "kind": "llm_prior_uniform_fallback",
                    "sim_id": sim_id,
                    "prior_source": "api_call_failed",
                    "mapping_mode": "uniform_api_fallback",
                    "logic": logic,
                    "partial_strategy": partial_strategy,
                    "candidate_actions": candidate_actions,
                    "priors": u,
                    "ranked_priors": self._ranked_priors(u),
                }
            )
            self._log_prior_event(
                sim_id,
                "api_call_failed",
                mapping_mode="uniform_api_fallback",
                detail=self._format_ranked_priors(u),
            )
            return u

        by_name: dict[str, int] = {}
        for item in parsed.scores:
            by_name[str(item.tactic_name)] = item.value

        scores: dict[str, int] = {}
        for a in candidate_actions:
            scores[a] = by_name.get(a, 0)

        mapping_mode, priors, reason = self._thresholded_softmax(scores)
        self._memory[partial_strategy] = (mapping_mode, priors)
        self.last_prior_source = "api_call"
        self.last_mapping_mode = mapping_mode
        ranked = self._ranked_priors(priors)
        self._append_qa_log(
            {
                "kind": "llm_prior_scores",
                "sim_id": sim_id,
                "prior_source": "api_call",
                "reason": reason,
                "mapping_mode": mapping_mode,
                "logic": logic,
                "partial_strategy": partial_strategy,
                "candidate_actions": candidate_actions,
                "scores": scores,
                "priors": priors,
                "ranked_priors": ranked,
            }
        )
        self._log_prior_event(
            sim_id,
            "api_call",
            mapping_mode=mapping_mode,
            detail=(
                f"reason={reason} {self._format_ranked_priors(priors)}"
                if reason
                else self._format_ranked_priors(priors)
            ),
        )
        return {a: priors[a] for a in candidate_actions}
