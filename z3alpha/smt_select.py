"""
PWC (pairwise comparison) based algorithm selector for Z3 tactic strategies.

Training API. Inference (feature extraction, ranking) is in smt_select_infer.py.

Approach adapted from SMT-Select (syntactic/PWC variant):
  https://github.com/JohnLyu2/smt-select
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from z3alpha.smt_select_infer import (
    FEATURE_NAMES,
    PairwiseSelector,
    bench_feature_vector,
)

log = logging.getLogger(__name__)

_PERF_DIFF_THRESHOLD = 0.1

__all__ = [
    "FEATURE_NAMES",
    "PairwiseSelector",
    "bench_feature_vector",
    "train_pwc_selector",
]


def _par_n(solved: bool, time_s: float, timeout: float, n: float) -> float:
    return time_s if solved else n * timeout


def train_pwc_selector(
    shortlist: list[tuple[str, list[tuple[bool, float, str]]]],
    bench_paths: list[str | Path],
    timeout: float,
    random_seed: int = 42,
    precomputed_features: Optional[dict[str, np.ndarray]] = None,
    perf_diff_threshold: int | float = _PERF_DIFF_THRESHOLD,
    model_type: str = "svm",
    weight_type: str = "par2",
    n_estimators: int = 100,
    max_depth: int = 6,
) -> PairwiseSelector:
    """Train a PWC selector from Stage 1 results.

    Args:
        shortlist: list of (strategy_str, per_bench_results) pairs from Stage 1,
            where per_bench_results is a list of (solved, time_s, status) tuples
            in the same order as bench_paths.
        bench_paths: benchmark file paths (used as keys into precomputed_features).
        timeout: solver timeout used during Stage 1 evaluation (seconds).
        random_seed: used for SVM and tie-breaking.
        precomputed_features: optional dict mapping path str -> feature vector.
            If provided, feature extraction from disk is skipped entirely.

    Returns:
        A trained PairwiseSelector ready for serialization and inference.
    """
    strategies = [s for s, _ in shortlist]
    bench_results = {s: r for s, r in shortlist}

    k = len(strategies)
    if k == 1:
        log.warning("Only 1 strategy in shortlist; selector always returns it")
        if precomputed_features:
            sample = next(iter(precomputed_features.values()), np.zeros(1))
        else:
            feats = [bench_feature_vector(p) for p in bench_paths]
            sample = next((f for f in feats if f is not None), np.zeros(1))
        scaler = StandardScaler()
        scaler.fit(sample.reshape(1, -1))
        return PairwiseSelector(
            strategies=strategies,
            model_matrix=np.empty((1, 1), dtype=object),
            scaler=scaler,
            fallback_idx=0,
        )
    assert k >= 2

    raw_features: list[np.ndarray] = []
    valid_indices: list[int] = []
    for idx, path in enumerate(bench_paths):
        path_key = str(path)
        if precomputed_features is not None:
            feat = precomputed_features.get(path_key)
        else:
            feat = bench_feature_vector(path)
        if feat is not None:
            raw_features.append(feat)
            valid_indices.append(idx)

    if not raw_features:
        raise ValueError("Feature extraction failed for all benchmarks")

    X = np.array(raw_features, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info(f"Features extracted for {len(valid_indices)}/{len(bench_paths)} benchmarks")

    par2 = np.zeros((len(valid_indices), k), dtype=float)
    for si, strat in enumerate(strategies):
        results = bench_results[strat]
        for row, bi in enumerate(valid_indices):
            solved, time_s, _ = results[bi]
            par2[row, si] = _par_n(solved, time_s, timeout, 2)

    fallback_idx = int(np.argmin(par2.mean(axis=0)))
    log.info(f"SBS fallback: strategy {fallback_idx} ({strategies[fallback_idx][:60]}...)")

    w_n = 10 if weight_type == "par10" else 2
    if w_n != 2:
        w_matrix = np.zeros_like(par2)
        for si, strat in enumerate(strategies):
            results = bench_results[strat]
            for row, bi in enumerate(valid_indices):
                solved, time_s, _ = results[bi]
                w_matrix[row, si] = _par_n(solved, time_s, timeout, w_n)
    else:
        w_matrix = par2

    model_matrix = np.empty((k, k), dtype=object)
    for i in range(k):
        for j in range(i + 1, k):
            diff = w_matrix[:, i] - w_matrix[:, j]
            labels = (diff < 0).astype(int)
            costs = np.abs(diff)
            mask = costs > perf_diff_threshold

            if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
                model = DummyClassifier(strategy="most_frequent")
                fit_X = X_scaled[mask] if mask.sum() > 0 else X_scaled
                fit_y = labels[mask] if mask.sum() > 0 else labels
                model.fit(fit_X, fit_y)
            elif model_type == "xgboost":
                model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=random_seed, verbosity=0, nthread=4)
                model.fit(X_scaled[mask], labels[mask], sample_weight=costs[mask])
            else:
                model = SVC(kernel="rbf", random_state=random_seed)
                model.fit(X_scaled[mask], labels[mask], sample_weight=costs[mask])

            model_matrix[i, j] = model
            log.debug(f"Trained model ({i},{j}): {mask.sum()} samples, "
                      f"type={type(model).__name__}")

    log.info(f"Trained {k*(k-1)//2} pairwise models for {k} strategies")
    return PairwiseSelector(
        strategies=strategies,
        model_matrix=model_matrix,
        scaler=scaler,
        fallback_idx=fallback_idx,
        par2_scores=par2.mean(axis=0),
    )
