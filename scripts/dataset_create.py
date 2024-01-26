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
    parser.add_argument('--benchmark_dir', type=str, required=True, help='Benchmark directory')
    parser.add_argument('--dataset_dir', type=str, required=True, help='The directory that stores the created dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    benchmark_dir = args.benchmark_dir
    assert os.path.exists(benchmark_dir), 'The specified benchmark folder does not exist!'
    
    all_files = []
    for file in sorted(list(pathlib.Path(benchmark_dir).rglob("*.smt2"))):
        all_files.append(str(file))
    all_file_size = len(all_files)
    random.shuffle(all_files)

    train_size, valid_size = list(map(int, args.split_size.split(' ')))
    assert train_size + valid_size <= all_file_size, 'Sum of train&valid sizes should be smaller than the total smt2 file size'
    test_size = all_file_size - train_size - valid_size

    dataset_dir = args.dataset_dir
    assert(not os.path.exists(dataset_dir))
    os.mkdir(dataset_dir)
    train_dir = os.path.join(dataset_dir, 'train1')
    valid_dir = os.path.join(dataset_dir, 'train2')
    test_dir = os.path.join(dataset_dir, 'test')

    os.mkdir(train_dir)
    os.mkdir(valid_dir)
    os.mkdir(test_dir)

    create_dir(train_dir, all_files[:train_size])
    create_dir(valid_dir, all_files[train_size:train_size+valid_size])
    create_dir(test_dir, all_files[train_size+valid_size:train_size+valid_size+test_size])

# sample usage: python dataset_create.py --split_size "250 500" --benchmark_dir smtlib/QF_S/2019-Jiang/slog --dataset_dir smtlib/QF_S/2019-Jiang/slog_exp
if __name__ == '__main__':
    main()