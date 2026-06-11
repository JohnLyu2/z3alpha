"""
PWC selector inference: feature extraction and pairwise strategy ranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from z3alpha.smtlib_features import extract_features, SMTLIB_SYMBOLS
except ImportError:
    from smtlib_features import extract_features, SMTLIB_SYMBOLS  # type: ignore[no-redef]

log = logging.getLogger(__name__)

_STRUCTURAL_NAMES = [
    "assertsCount",
    "declareFunCount",
    "declareConstCount",
    "declareSortCount",
    "defineFunCount",
    "defineFunRecCount",
    "constantFunCount",
    "defineSortCount",
    "declareDatatypeCount",
    "maxTermDepth",
]

# Comparison operator pairs merged into a single summed feature.
# Each entry is (keep_sym, drop_sym, merged_name) using SMTLIB_SYMBOLS names.
_ANTISYM_PAIRS: list[tuple[str, str, str]] = [
    ("<",       ">",       "<>"),
    ("<=",      ">=",      "<=>"),
    ("bvult",   "bvugt",   "bvult_bvugt"),
    ("bvule",   "bvuge",   "bvule_bvuge"),
    ("bvslt",   "bvsgt",   "bvslt_bvsgt"),
    ("bvsle",   "bvsge",   "bvsle_bvsge"),
    ("fp.lt",   "fp.gt",   "fp.lt_fp.gt"),
    ("fp.leq",  "fp.geq",  "fp.leq_fp.geq"),
]

# Build index-level merge map from SMTLIB_SYMBOLS once at import time.
# _sym_offset: number of structural features before the symbol block.
_sym_offset = len(_STRUCTURAL_NAMES)
_keep_indices: list[int] = []
_drop_indices: set[int] = set()
_keep_to_merged: dict[int, str] = {}

for _keep_sym, _drop_sym, _merged in _ANTISYM_PAIRS:
    _ki = _sym_offset + SMTLIB_SYMBOLS.index(_keep_sym)
    _di = _sym_offset + SMTLIB_SYMBOLS.index(_drop_sym)
    _keep_indices.append(_ki)
    _drop_indices.add(_di)
    _keep_to_merged[_ki] = f"sym_{_merged}"

_raw_names: list[str] = _STRUCTURAL_NAMES + [f"sym_{s}" for s in SMTLIB_SYMBOLS]
FEATURE_NAMES: list[str] = [
    _keep_to_merged.get(i, name)
    for i, name in enumerate(_raw_names)
    if i not in _drop_indices
]

# Boolean mask for dropping the second element of each pair from a raw vector.
_keep_mask = np.array([i not in _drop_indices for i in range(len(_raw_names))], dtype=bool)


def bench_feature_vector(path: str | Path) -> Optional[np.ndarray]:
    """Extract a flat feature vector from an SMT-LIB 2 benchmark.

    For incremental benchmarks (multiple check-sat calls), structural counts
    are summed and max_term_depth is taken as the maximum across all queries.
    Returns None if extraction fails.
    """
    try:
        entries = extract_features(path)
    except Exception:
        log.debug(f"Feature extraction failed for {path}", exc_info=True)
        return None

    queries = entries[:-1]  # last entry is benchmark-level metadata
    if not queries:
        return None

    vec = [
        sum(q["assertsCount"]          for q in queries),
        sum(q["declareFunCount"]       for q in queries),
        sum(q["declareConstCount"]     for q in queries),
        sum(q["declareSortCount"]      for q in queries),
        sum(q["defineFunCount"]        for q in queries),
        sum(q["defineFunRecCount"]     for q in queries),
        sum(q["constantFunCount"]      for q in queries),
        sum(q["defineSortCount"]       for q in queries),
        sum(q["declareDatatypeCount"]  for q in queries),
        max(q["maxTermDepth"]          for q in queries),
    ]
    freq = [0] * len(queries[0]["symbolFrequency"])
    for q in queries:
        for i, v in enumerate(q["symbolFrequency"]):
            freq[i] += v
    vec.extend(freq)

    raw = np.array(vec, dtype=float)
    for ki, di in zip(_keep_indices, sorted(_drop_indices)):
        raw[ki] += raw[di]
    return raw[_keep_mask]


@dataclass
class PairwiseSelector:
    """Trained pairwise comparison strategy selector."""

    strategies: list[str]
    model_matrix: np.ndarray
    scaler: StandardScaler
    fallback_idx: int
    par2_scores: Optional[np.ndarray] = None

    def select(self, path: str | Path) -> str:
        """Return the strategy string predicted to work best on this benchmark."""
        if len(self.strategies) == 1:
            return self.strategies[0]

        raw = bench_feature_vector(path)
        if raw is None:
            log.warning(f"Feature extraction failed for {path}; using fallback strategy")
            return self.strategies[self.fallback_idx]

        return self.select_vec(raw)

    def _rank_vec(self, raw: np.ndarray) -> list[str]:
        """Return strategies ranked by vote count (desc), PAR-2 as tie-breaker."""
        x = self.scaler.transform(raw.reshape(1, -1))
        k = len(self.strategies)
        votes = [0] * k
        for i in range(k):
            for j in range(i + 1, k):
                pred = int(self.model_matrix[i, j].predict(x)[0])
                if pred == 1:
                    votes[i] += 1
                else:
                    votes[j] += 1
        if self.par2_scores is not None:
            order = sorted(range(k), key=lambda i: (-votes[i], self.par2_scores[i]))
        else:
            order = sorted(range(k), key=lambda i: -votes[i])
        return [self.strategies[i] for i in order]

    def select_vec(self, raw: np.ndarray) -> str:
        """Return the top-ranked strategy given a precomputed feature vector."""
        if len(self.strategies) == 1:
            return self.strategies[0]
        return self._rank_vec(raw)[0]

    def rank_all_vec(self, raw: np.ndarray) -> list[str]:
        """Return all strategies in ranked order given a precomputed feature vector."""
        if len(self.strategies) == 1:
            return list(self.strategies)
        return self._rank_vec(raw)

    def rank_all(self, path: str | Path) -> list[str]:
        """Return all strategies in ranked order for a benchmark file."""
        raw = bench_feature_vector(path)
        if raw is None:
            log.warning(f"Feature extraction failed for {path}; using fallback ordering")
            fallback = list(self.strategies)
            fallback.insert(0, fallback.pop(self.fallback_idx))
            return fallback
        return self.rank_all_vec(raw)

    def select_top2_vec(self, raw: np.ndarray) -> tuple[str, str]:
        """Return (rank-1, rank-2) strategies given a precomputed feature vector."""
        if len(self.strategies) == 1:
            s = self.strategies[0]
            return s, s
        ranked = self._rank_vec(raw)
        return ranked[0], ranked[1]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
        log.info(f"Selector saved to {path}")

    @staticmethod
    def load(path: str | Path) -> PairwiseSelector:
        return joblib.load(path)
