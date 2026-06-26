"""Unit tests for stratified benchmark prescreen helpers."""

import sys
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from stratified_dataset_lib import (  # noqa: E402
    assign_buckets,
    bucket_row,
    sample_per_bucket,
)


class TestBucketRow(unittest.TestCase):
    def test_easy(self) -> None:
        self.assertEqual(bucket_row(True, 0.5, 10.0), "easy")

    def test_easy_at_cutoff(self) -> None:
        self.assertEqual(bucket_row(True, 1.0, 10.0), "easy")

    def test_medium(self) -> None:
        self.assertEqual(bucket_row(True, 5.0, 10.0), "medium")

    def test_medium_just_above_easy(self) -> None:
        self.assertEqual(bucket_row(True, 1.01, 10.0), "medium")

    def test_medium_at_timeout(self) -> None:
        self.assertEqual(bucket_row(True, 10.0, 10.0), "medium")

    def test_hard_unsolved(self) -> None:
        self.assertEqual(bucket_row(False, 10.0, 10.0), "hard")

    def test_hard_over_timeout(self) -> None:
        self.assertEqual(bucket_row(True, 10.01, 10.0), "hard")


class TestSamplePerBucket(unittest.TestCase):
    def _rows(self) -> list[dict]:
        base = [
            {"path": f"easy{i}.smt2", "solved": True, "time_s": 0.1, "bucket": "easy"}
            for i in range(5)
        ]
        base += [
            {"path": f"med{i}.smt2", "solved": True, "time_s": 5.0, "bucket": "medium"}
            for i in range(5)
        ]
        base += [
            {"path": f"hard{i}.smt2", "solved": False, "time_s": 10.0, "bucket": "hard"}
            for i in range(5)
        ]
        return base

    def test_respects_per_bucket_cap(self) -> None:
        sampled, warnings = sample_per_bucket(self._rows(), per_bucket=3, seed=0)
        self.assertEqual(len(sampled), 9)
        self.assertEqual([], warnings)
        by_bucket: dict[str, int] = {}
        for r in sampled:
            by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
        self.assertEqual(by_bucket, {"easy": 3, "medium": 3, "hard": 3})

    def test_warning_when_pool_small(self) -> None:
        rows = self._rows()[:2]
        rows[0]["bucket"] = "easy"
        rows[1]["bucket"] = "easy"
        sampled, warnings = sample_per_bucket(rows, per_bucket=15, seed=0)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(len(warnings), 3)

    def test_reproducible_with_seed(self) -> None:
        a, _ = sample_per_bucket(self._rows(), per_bucket=2, seed=99)
        b, _ = sample_per_bucket(self._rows(), per_bucket=2, seed=99)
        self.assertEqual([r["path"] for r in a], [r["path"] for r in b])


class TestAssignBuckets(unittest.TestCase):
    def test_assign(self) -> None:
        rows = [{"solved": True, "time_s": 0.5}, {"solved": False, "time_s": 10.0}]
        assign_buckets(rows, 10.0)
        self.assertEqual(rows[0]["bucket"], "easy")
        self.assertEqual(rows[1]["bucket"], "hard")


if __name__ == "__main__":
    unittest.main()
