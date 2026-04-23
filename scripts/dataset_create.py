import argparse
import csv
import logging
import random
from pathlib import Path

from z3alpha.config import setup_logging

log = logging.getLogger(__name__)


def create_dir(target_dir, files):
    counter = 0
    target_dir = Path(target_dir)
    log.info("Creating symlinks in %s (%s files)...", target_dir, len(files))
    for source_filepath in files:
        source_path = Path(source_filepath).resolve()  # Get absolute path
        target_filepath = target_dir / f"benchmark{counter}.smt2"
        target_filepath.symlink_to(source_path)
        counter += 1
    log.info("Created %s symlinks in %s", counter, target_dir)


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
    setup_logging()

    log.info("%s", "=" * 60)
    log.info("Dataset Creation Script")
    log.info("%s", "=" * 60)
    log.info("Split sizes: %s", args.split_size)
    log.info("Dataset directory: %s", args.dataset_dir)
    log.info("Random seed: %s", args.seed)
    if args.csv_file:
        log.info("CSV file: %s", args.csv_file)
        log.info("Prefix: %s", args.prefix)
    else:
        log.info("Benchmark directory: %s", args.benchmark_dir)
    log.info("%s", "=" * 60)

    random.seed(args.seed)

    all_files = []
    if args.csv_file:
        # If CSV is provided, read paths from CSV and prepend prefix
        csv_path = Path(args.csv_file)
        assert csv_path.exists(), f"The specified CSV file does not exist: {args.csv_file}"
        assert args.prefix is not None, "--prefix is required when --csv_file is provided"
        
        log.info("Reading file paths from CSV: %s", csv_path)
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
                        log.warning("File not found: %s", full_path)
        log.info("Found %s files from CSV", len(all_files))
        if missing_files > 0:
            log.warning("%s files from CSV were not found", missing_files)
    else:
        # Original behavior: scan benchmark_dir for all .smt2 files
        assert args.benchmark_dir is not None, "--benchmark_dir is required when --csv_file is not provided"
        benchmark_dir = Path(args.benchmark_dir)
        assert benchmark_dir.exists(), "The specified benchmark folder does not exist!"
        log.info("Scanning benchmark directory: %s", benchmark_dir)
        for file in sorted(benchmark_dir.rglob("*.smt2")):
            all_files.append(str(file))
        log.info("Found %s .smt2 files", len(all_files))

    all_file_size = len(all_files)
    log.info("")
    log.info("Total files to process: %s", all_file_size)
    log.info("Shuffling files...")
    random.shuffle(all_files)

    train_size, valid_size = list(map(int, args.split_size.split(" ")))
    assert train_size + valid_size <= all_file_size, (
        "Sum of train&valid sizes should be smaller than the total smt2 file size"
    )
    test_size = all_file_size - train_size - valid_size

    log.info("")
    log.info("Dataset split:")
    log.info("  Train: %s files", train_size)
    log.info("  Valid: %s files", valid_size)
    log.info("  Test:  %s files", test_size)

    dataset_dir = Path(args.dataset_dir)
    log.info("")
    log.info("Creating dataset directory: %s", dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_dir = dataset_dir / "train1"
    valid_dir = dataset_dir / "train2"
    test_dir = dataset_dir / "test"

    log.info("Creating subdirectories...")
    train_dir.mkdir()
    valid_dir.mkdir()
    test_dir.mkdir()

    log.info("")
    log.info("Creating symlinks...")
    create_dir(train_dir, all_files[:train_size])
    create_dir(valid_dir, all_files[train_size : train_size + valid_size])
    create_dir(
        test_dir,
        all_files[train_size + valid_size : train_size + valid_size + test_size],
    )
    
    log.info("")
    log.info("%s", "=" * 60)
    log.info("Dataset creation completed successfully!")
    log.info("Dataset location: %s", dataset_dir)
    log.info("%s", "=" * 60)


# sample usage: python dataset_create.py --split_size "250 500" --benchmark_dir smtlib/QF_S/2019-Jiang/slog --dataset_dir smtlib/QF_S/2019-Jiang/slog_exp
# sample usage with CSV: python dataset_create.py --split_size "250 500" --dataset_dir dataset --csv_file data/hard/qfnia_2unsolve.csv --prefix smtlib
if __name__ == "__main__":
    main()
