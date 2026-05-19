"""
PWC (pairwise comparison) based algorithm selector for Z3 tactic strategies.

Trains one binary SVM per pair of shortlisted strategies. At inference time,
features are extracted from the benchmark using smtlib_features.py, pairwise
votes are collected, and the strategy with the most votes is selected.

Approach adapted from SMT-Select (syntactic/PWC variant):
  https://github.com/JohnLyu2/smt-select
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from z3alpha.smtlib_features import extract_features

log = logging.getLogger(__name__)

# Ignore pairwise samples where the performance gap is smaller than this (seconds)
_PERF_DIFF_THRESHOLD = 0.1


# ─── Feature extraction ───────────────────────────────────────────────────────

def _bench_features(path: str | Path) -> Optional[np.ndarray]:
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
        sum(q["normalizedSize"]        for q in queries),
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
    # Sum symbol frequencies across all queries
    freq = [0] * len(queries[0]["symbolFrequency"])
    for q in queries:
        for i, v in enumerate(q["symbolFrequency"]):
            freq[i] += v
    vec.extend(freq)
    return np.array(vec, dtype=float)


# ─── Selector ─────────────────────────────────────────────────────────────────

@dataclass
class PwcSelector:
    """Trained pairwise comparison strategy selector.

    Attributes:
        strategies: shortlisted strategy strings (index matches model_matrix).
        model_matrix: upper-triangular K×K array; model_matrix[i,j] (i<j)
            predicts 1 if strategy i is better than j, 0 otherwise.
        scaler: fitted StandardScaler used during training.
        fallback_idx: index of the single-best-strategy fallback.
    """

    strategies: list[str]
    model_matrix: np.ndarray
    scaler: StandardScaler
    fallback_idx: int

    def select(self, path: str | Path) -> str:
        """Return the strategy string predicted to work best on this benchmark."""
        raw = _bench_features(path)
        if raw is None:
            log.warning(f"Feature extraction failed for {path}; using fallback strategy")
            return self.strategies[self.fallback_idx]

        if len(self.strategies) == 1:
            return self.strategies[0]

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

        return self.strategies[int(np.argmax(votes))]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
        log.info(f"Selector saved to {path}")

    @staticmethod
    def load(path: str | Path) -> PwcSelector:
        return joblib.load(path)


# ─── Training ─────────────────────────────────────────────────────────────────

def _par2(solved: bool, time_s: float, timeout: float) -> float:
    return time_s if solved else 2.0 * timeout


def train_pwc_selector(
    strategies: list[str],
    bench_paths: list[str | Path],
    bench_results: dict[str, list[tuple[bool, float, str]]],
    timeout: float,
    random_seed: int = 42,
) -> PwcSelector:
    """Train a PWC selector from Stage 1 results.

    Args:
        strategies: shortlisted strategy strings.
        bench_paths: benchmark file paths (same order as bench_results lists).
        bench_results: maps strategy_str → list of (solved, time_s, status),
            one entry per benchmark in the same order as bench_paths.
        timeout: solver timeout used during Stage 1 evaluation (seconds).
        random_seed: used for SVM and tie-breaking.

    Returns:
        A trained PwcSelector ready for serialization and inference.
    """
    k = len(strategies)
    if k == 1:
        log.warning("Only 1 strategy in shortlist; selector always returns it")
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 1)))  # dummy fit
        return PwcSelector(
            strategies=strategies,
            model_matrix=np.empty((1, 1), dtype=object),
            scaler=scaler,
            fallback_idx=0,
        )
    assert k >= 2

    # ── Extract features ──────────────────────────────────────────────────
    raw_features: list[np.ndarray] = []
    valid_indices: list[int] = []
    for idx, path in enumerate(bench_paths):
        feat = _bench_features(path)
        if feat is not None:
            raw_features.append(feat)
            valid_indices.append(idx)

    if not raw_features:
        raise ValueError("Feature extraction failed for all benchmarks")

    X = np.array(raw_features, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info(f"Features extracted for {len(valid_indices)}/{len(bench_paths)} benchmarks")

    # ── PAR-2 matrix (n_valid × k) ────────────────────────────────────────
    par2 = np.zeros((len(valid_indices), k), dtype=float)
    for si, strat in enumerate(strategies):
        results = bench_results[strat]
        for row, bi in enumerate(valid_indices):
            solved, time_s, _ = results[bi]
            par2[row, si] = _par2(solved, time_s, timeout)

    fallback_idx = int(np.argmin(par2.mean(axis=0)))
    log.info(f"SBS fallback: strategy {fallback_idx} ({strategies[fallback_idx][:60]}...)")

    # ── Train pairwise models ─────────────────────────────────────────────
    model_matrix = np.empty((k, k), dtype=object)
    for i in range(k):
        for j in range(i + 1, k):
            diff = par2[:, i] - par2[:, j]
            labels = (diff < 0).astype(int)   # 1 = strategy i is better
            costs = np.abs(diff)
            mask = costs > _PERF_DIFF_THRESHOLD

            if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
                model = DummyClassifier(strategy="most_frequent")
                fit_X = X_scaled[mask] if mask.sum() > 0 else X_scaled
                fit_y = labels[mask] if mask.sum() > 0 else labels
                model.fit(fit_X, fit_y)
            else:
                model = SVC(kernel="rbf", random_state=random_seed)
                model.fit(X_scaled[mask], labels[mask], sample_weight=costs[mask])

            model_matrix[i, j] = model
            log.debug(f"Trained model ({i},{j}): {mask.sum()} samples, "
                      f"type={type(model).__name__}")

    log.info(f"Trained {k*(k-1)//2} pairwise models for {k} strategies")
    return PwcSelector(
        strategies=strategies,
        model_matrix=model_matrix,
        scaler=scaler,
        fallback_idx=fallback_idx,
    )
