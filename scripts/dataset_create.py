import argparse
import random
import shutil
from pathlib import Path


def create_dir(target_dir, files):
    counter = 0
    target_dir = Path(target_dir)
    for source_filepath in files:
        source_path = Path(source_filepath).resolve()  # Get absolute path
        target_filepath = target_dir / f"benchmark{counter}.smt2"
        target_filepath.symlink_to(source_path)
        counter += 1


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
        "--benchmark_dir", type=str, required=True, help="Benchmark directory"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory that stores the created dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    benchmark_dir = Path(args.benchmark_dir)
    assert benchmark_dir.exists(), "The specified benchmark folder does not exist!"

    all_files = []
    for file in sorted(benchmark_dir.rglob("*.smt2")):
        all_files.append(str(file))
    all_file_size = len(all_files)
    random.shuffle(all_files)

    train_size, valid_size = list(map(int, args.split_size.split(" ")))
    assert train_size + valid_size <= all_file_size, (
        "Sum of train&valid sizes should be smaller than the total smt2 file size"
    )
    test_size = all_file_size - train_size - valid_size

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_dir = dataset_dir / "train1"
    valid_dir = dataset_dir / "train2"
    test_dir = dataset_dir / "test"

    train_dir.mkdir()
    valid_dir.mkdir()
    test_dir.mkdir()

    create_dir(train_dir, all_files[:train_size])
    create_dir(valid_dir, all_files[train_size : train_size + valid_size])
    create_dir(
        test_dir,
        all_files[train_size + valid_size : train_size + valid_size + test_size],
    )


# sample usage: python dataset_create.py --split_size "250 500" --benchmark_dir smtlib/QF_S/2019-Jiang/slog --dataset_dir smtlib/QF_S/2019-Jiang/slog_exp
if __name__ == "__main__":
    main()
