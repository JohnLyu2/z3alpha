"""PWC (pairwise comparison) algorithm selection for Z3 tactic strategies.

Adapted from SMT-Select (syntactic/PWC variant): https://github.com/JohnLyu2/smt-select

Submodules:
- ``z3alpha.smt_select.features``: SMT-LIB benchmark feature extraction
  (Python re-implementation of klhm).
- ``z3alpha.smt_select.infer``: feature vectors and ``PairwiseSelector`` for
  inference/ranking.
- ``z3alpha.smt_select.train``: ``train_pwc_selector`` for training a selector
  from Stage 1 results.
"""

from z3alpha.smt_select.infer import (
    FEATURE_NAMES,
    PairwiseSelector,
    bench_feature_vector,
)
from z3alpha.smt_select.train import train_pwc_selector

__all__ = [
    "FEATURE_NAMES",
    "PairwiseSelector",
    "bench_feature_vector",
    "train_pwc_selector",
]
