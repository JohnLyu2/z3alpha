#!/usr/bin/env python3
import argparse
import random
import shutil
import csv
from pathlib import Path

def create_symlink_structure(target_root, source_files, source_root, benchmark_root_name):
    """
    Create symlinks in target_root preserving relative paths from source_root.
    Returns a dictionary mapping CSV benchmark keys to symlink paths.
    """
    target_root = Path(target_root).resolve()
    source_root = Path(source_root).resolve()
    symlink_map = {}

    for src in source_files:
        src_path = Path(src).resolve()
        rel_path = src_path.relative_to(source_root)
        dst_path = target_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        dst_path.symlink_to(src_path)
        # Key as in original CSV: <benchmark_root_name>/<relative path>
        csv_key = f"{benchmark_root_name}/{rel_path.as_posix()}"
        symlink_map[csv_key] = str(dst_path)  # <- DO NOT resolve, keep the symlink path
    return symlink_map

def filter_csv(original_csv_path, output_csv_path, symlink_map):
    """
    Filter the original CSV to only include benchmarks in symlink_map.
    Replaces benchmark path with symlinked path.
    """
    with open(original_csv_path, newline='') as f_in, open(output_csv_path, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        skipped = 0
        for row in reader:
            benchmark_key = row['benchmark']
            if benchmark_key in symlink_map:
                row['benchmark'] = symlink_map[benchmark_key]  # This now points to the symlink
                writer.writerow(row)
                count += 1
            else:
                skipped += 1
                print(f"Skipped CSV row (no match found): {benchmark_key}")
        print(f"Filtered {count} rows written to {output_csv_path}")
        if skipped > 0:
            print(f"Skipped {skipped} CSV rows due to unmatched benchmarks.")

def main():
    parser = argparse.ArgumentParser(description="Split SMT benchmarks into train/test sets with CSV mapping of solver results.")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Fraction of files to use for training (default: 0.8)")
    parser.add_argument("--benchmark_dir", type=str, required=True,
                        help="Root benchmark directory containing .smt2 files")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory where train/test folders will be created")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="CSV file with all solver results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    benchmark_dir = Path(args.benchmark_dir).resolve()
    assert benchmark_dir.exists(), f"Benchmark directory {benchmark_dir} does not exist!"

    # Collect all .smt2 files
    all_files = sorted(benchmark_dir.rglob("*.smt2"))
    if not all_files:
        raise RuntimeError(f"No .smt2 files found in {benchmark_dir}")
    random.shuffle(all_files)

    print(f"Found {len(all_files)} total .smt2 files in benchmark_dir")
    print(f"Sample benchmark files: {[str(f.relative_to(benchmark_dir)) for f in all_files[:5]]}")

    # Split files
    train_count = int(len(all_files) * args.split_ratio)
    train_files = all_files[:train_count]
    test_files = all_files[train_count:]
    print(f"Train count: {len(train_files)}, Test count: {len(test_files)}")

    # Dataset directories
    dataset_root = Path(args.dataset_dir).resolve()
    benchmark_root_name = benchmark_dir.name
    train_root = dataset_root / benchmark_root_name / "train"
    test_root  = dataset_root / benchmark_root_name / "test"

    # Clean existing folders
    if train_root.exists():
        shutil.rmtree(train_root)
    if test_root.exists():
        shutil.rmtree(test_root)

    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    # Create symlinks and get symlink map
    train_symlink_map = create_symlink_structure(train_root, train_files, benchmark_dir, benchmark_root_name)
    test_symlink_map  = create_symlink_structure(test_root, test_files, benchmark_dir, benchmark_root_name)

    print(f"Sample train symlink map keys: {list(train_symlink_map.keys())[:5]}")
    print(f"Sample test symlink map keys: {list(test_symlink_map.keys())[:5]}")

    # Generate CSVs using symlink paths
    filter_csv(args.results_csv, train_root / "train.csv", train_symlink_map)
    filter_csv(args.results_csv, test_root / "test.csv", test_symlink_map)

    print(f"✅ Train/test splits and CSVs created under {dataset_root / benchmark_root_name}")
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
    print(f"Train CSV: {train_root / 'train.csv'}")
    print(f"Test CSV: {test_root / 'test.csv'}")

if __name__ == "__main__":
    main()
