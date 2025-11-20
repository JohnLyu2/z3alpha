#!/usr/bin/env python3
"""
Dataset Splitter for SMT Benchmarks

This script splits .smt2 benchmark files into train and test sets.
It preserves the original folder hierarchy and creates symbolic links
in the new train/ and test/ directories.

Example:
  Original: QF_SLIA/family/benchmark.smt2
  New:
    QF_SLIA/train/family/benchmark.smt2
    QF_SLIA/test/family/benchmark.smt2

Usage:
  python dataset_split.py --split_ratio 0.8 \
    --benchmark_dir QF_SLIA \
    --dataset_dir QF_SLIA \
    --seed 42
    
    
python '/home/jchen688/projects/def-vganesh/jchen688/github/z3alpha/JHC/AlgoSelect/scripts/dataset_create.py' \
--split_ratio 0.8 \
--benchmark_dir /home/jchen688/scratch/jchen688/SMTCOMP24/non-incremental/QF_UFDT \
--dataset_dir /home/jchen688/projects/def-vganesh/jchen688/github/z3alpha/JHC/AlgoSelect/benchmarks/SMTCOMP24
"""

#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

def create_symlink_structure(target_root, source_files, source_root):
    """Create symlinks in target_root preserving relative paths from source_root."""
    source_root = source_root.resolve()
    for src in source_files:
        src_path = Path(src).resolve()
        rel_path = src_path.relative_to(source_root)
        dst_path = target_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst_path.symlink_to(src_path)
        except FileExistsError:
            dst_path.unlink()
            dst_path.symlink_to(src_path)

def main():
    parser = argparse.ArgumentParser(description="Split SMT benchmarks into train/test sets with preserved folder structure.")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Fraction of files to use for training (default: 0.8)")
    parser.add_argument("--benchmark_dir", type=str, required=True,
                        help="Root benchmark directory containing .smt2 files")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory where train/test folders will be created")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    benchmark_dir = Path(args.benchmark_dir).resolve()
    assert benchmark_dir.exists(), f"Benchmark directory {benchmark_dir} does not exist!"

    all_files = sorted(benchmark_dir.rglob("*.smt2"))
    if not all_files:
        raise RuntimeError(f"No .smt2 files found in {benchmark_dir}")

    random.shuffle(all_files)

    train_count = int(len(all_files) * args.split_ratio)
    train_files = all_files[:train_count]
    test_files = all_files[train_count:]

    dataset_root = Path(args.dataset_dir).resolve()

    # Include the immediate benchmark_dir folder name in the new dataset path
    benchmark_parent_name = benchmark_dir.name
    train_root = dataset_root / benchmark_parent_name / "train"
    test_root  = dataset_root / benchmark_parent_name / "test"

    # Create symlinks preserving hierarchy
    create_symlink_structure(train_root, train_files, benchmark_dir)
    create_symlink_structure(test_root, test_files, benchmark_dir)

    print(f"✅ Created train/test splits under {dataset_root / benchmark_parent_name}")
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

if __name__ == "__main__":
    main()
