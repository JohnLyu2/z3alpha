import argparse
import os
import random
import shutil
import pathlib

def create_dir(target_dir, files):
    counter = 0
    for source_filepath in files:
        # file = pathlib.Path(source_filepath).name
        target_filepath = os.path.join(target_dir, f"benchmark{counter}.smt2")
        os.symlink(source_filepath, target_filepath)
        counter += 1

def main():
    parser = argparse.ArgumentParser(description='Create split dataset from files in the folder.')
    parser.add_argument('--split_size', type=str, required=True, help='Size of formulas that should go into train and valid, the rest goes into test')
    parser.add_argument('--benchmark_dir', type=str, required=True, help='Benchmark folder')
    parser.add_argument('--dataset_folder', type=str, required=True, help='the folder under the benchmark folder that stores the created dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    assert os.path.exists(args.benchmark_dir), 'The specified benchmark folder does not exist!'
    benchmark_dir = os.path.abspath(args.benchmark_dir)
    all_files = []
    for file in sorted(list(pathlib.Path(benchmark_dir).rglob("*.smt2"))):
        all_files.append(str(file))
    all_file_size = len(all_files)

    train_size, valid_size = list(map(int, args.split_size.split(' ')))
    assert train_size + valid_size <= all_file_size, 'Sum of train&valid sizes should be smaller than the total smt2 file size'
    test_size = all_file_size - train_size - valid_size

    dataset_folder = os.path.join(benchmark_dir, args.dataset_folder)
    assert(not os.path.exists(dataset_folder))
    os.mkdir(dataset_folder)
    train_dir = os.path.abspath(os.path.join(dataset_folder, 'train'))
    valid_dir = os.path.abspath(os.path.join(dataset_folder, 'valid'))
    test_dir = os.path.abspath(os.path.join(dataset_folder, 'test'))

    os.mkdir(train_dir)
    os.mkdir(valid_dir)
    os.mkdir(test_dir)

    random.shuffle(all_files)

    create_dir(train_dir, all_files[:train_size])
    create_dir(valid_dir, all_files[train_size:train_size+valid_size])
    create_dir(test_dir, all_files[train_size+valid_size:train_size+valid_size+test_size])

# sample usage: python dataset_create.py --split_size "250 500" --benchmark_dir /home/z52lu/smtlib/QF_S/2019-Jiang/slog --dataset_folder /home/z52lu/smtlib/QF_S/2019-Jiang/slog_exp
if __name__ == '__main__':
    main()