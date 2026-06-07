#!/usr/bin/env python3
"""
Pack data/smtcomp26/submission into z3alpha2_smtcomp26.tar.gz.

The archive is written next to the submission directory.  __pycache__
directories and .gitignore are excluded.  Executable bits on bin/z3 and
z3alpha2.py are preserved via tarfile.

Usage:
    python pack_submission.py
"""

import tarfile
import sys
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = REPO_ROOT / "data/smtcomp26/submission"
OUTPUT_TAR     = SUBMISSION_DIR.parent / "z3alpha2_smtcomp26.tar.gz"

EXCLUDE_NAMES = {"__pycache__", ".gitignore"}


def main():
    if not SUBMISSION_DIR.is_dir():
        sys.exit(f"Submission directory not found: {SUBMISSION_DIR}")

    files = sorted(
        p for p in SUBMISSION_DIR.rglob("*")
        if p.is_file() and not any(part in EXCLUDE_NAMES for part in p.parts)
    )

    with tarfile.open(OUTPUT_TAR, "w:gz") as tf:
        for f in files:
            arcname = f.relative_to(SUBMISSION_DIR)
            tf.add(f, arcname=str(arcname))
            print(f"  {arcname}")

    size_mb = OUTPUT_TAR.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUTPUT_TAR.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
