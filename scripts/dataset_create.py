import argparse
import csv
import random
import shutil
from pathlib import Path


def create_dir(target_dir, files):
    counter = 0
    target_dir = Path(target_dir)
    print(f"Creating symlinks in {target_dir} ({len(files)} files)...")
    for source_filepath in files:
        source_path = Path(source_filepath).resolve()  # Get absolute path
        target_filepath = target_dir / f"benchmark{counter}.smt2"
        target_filepath.symlink_to(source_path)
        counter += 1
    print(f"Created {counter} symlinks in {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create split dataset from files in the folder."
    )
    parser.add_argument(
        "--split_size",
        type=str,
        required=True,
        help="Size of formulas that should go into train and valid, the rest goes into test",
    )
    parser.add_argument(
        "--benchmark_dir", type=str, default=None, help="Benchmark directory (required when --csv_file is not provided)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory that stores the created dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Optional CSV file with smtlib_path column to specify files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix to prepend to CSV paths (only used when --csv_file is provided)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Creation Script")
    print("=" * 60)
    print(f"Split sizes: {args.split_size}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Random seed: {args.seed}")
    if args.csv_file:
        print(f"CSV file: {args.csv_file}")
        print(f"Prefix: {args.prefix}")
    else:
        print(f"Benchmark directory: {args.benchmark_dir}")
    print("=" * 60)

    random.seed(args.seed)

    all_files = []
    if args.csv_file:
        # If CSV is provided, read paths from CSV and prepend prefix
        csv_path = Path(args.csv_file)
        assert csv_path.exists(), f"The specified CSV file does not exist: {args.csv_file}"
        assert args.prefix is not None, "--prefix is required when --csv_file is provided"
        
        print(f"Reading file paths from CSV: {csv_path}")
        prefix = Path(args.prefix)
        missing_files = 0
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "smtlib_path" in row and row["smtlib_path"]:
                    full_path = prefix / row["smtlib_path"]
                    if full_path.exists():
                        all_files.append(str(full_path.resolve()))
                    else:
                        missing_files += 1
                        print(f"Warning: File not found: {full_path}")
        print(f"Found {len(all_files)} files from CSV")
        if missing_files > 0:
            print(f"Warning: {missing_files} files from CSV were not found")
    else:
        # Original behavior: scan benchmark_dir for all .smt2 files
        assert args.benchmark_dir is not None, "--benchmark_dir is required when --csv_file is not provided"
        benchmark_dir = Path(args.benchmark_dir)
        assert benchmark_dir.exists(), "The specified benchmark folder does not exist!"
        print(f"Scanning benchmark directory: {benchmark_dir}")
        for file in sorted(benchmark_dir.rglob("*.smt2")):
            all_files.append(str(file))
        print(f"Found {len(all_files)} .smt2 files")
    
    all_file_size = len(all_files)
    print(f"\nTotal files to process: {all_file_size}")
    print("Shuffling files...")
    random.shuffle(all_files)

    train_size, valid_size = list(map(int, args.split_size.split(" ")))
    assert train_size + valid_size <= all_file_size, (
        "Sum of train&valid sizes should be smaller than the total smt2 file size"
    )
    test_size = all_file_size - train_size - valid_size

    print(f"\nDataset split:")
    print(f"  Train: {train_size} files")
    print(f"  Valid: {valid_size} files")
    print(f"  Test:  {test_size} files")

    dataset_dir = Path(args.dataset_dir)
    print(f"\nCreating dataset directory: {dataset_dir}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_dir = dataset_dir / "train1"
    valid_dir = dataset_dir / "train2"
    test_dir = dataset_dir / "test"

    print(f"Creating subdirectories...")
    train_dir.mkdir()
    valid_dir.mkdir()
    test_dir.mkdir()

    print(f"\nCreating symlinks...")
    create_dir(train_dir, all_files[:train_size])
    create_dir(valid_dir, all_files[train_size : train_size + valid_size])
    create_dir(
        test_dir,
        all_files[train_size + valid_size : train_size + valid_size + test_size],
    )
    
    print("\n" + "=" * 60)
    print("Dataset creation completed successfully!")
    print(f"Dataset location: {dataset_dir}")
    print("=" * 60)


# sample usage: python dataset_create.py --split_size "250 500" --benchmark_dir smtlib/QF_S/2019-Jiang/slog --dataset_dir smtlib/QF_S/2019-Jiang/slog_exp
# sample usage with CSV: python dataset_create.py --split_size "250 500" --dataset_dir dataset --csv_file data/hard/qfnia_2unsolve.csv --prefix smtlib
if __name__ == "__main__":
    main()
