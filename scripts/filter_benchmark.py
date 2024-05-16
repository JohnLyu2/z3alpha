import os
import csv
import subprocess
from pathlib import Path

BENCH_DIR = "/home/z52lu/projects/def-vganesh/z52lu/smtlib24/non-incremental/QF_NIA"
DICT_PATH = "/home/z52lu/z3alpha/smtcomp24/results/QF_NIA/z3.csv"
CREATE_SET = True
TARGET_DIR = "/home/z52lu/z3alpha/smt24_bench/qfnia/gt16"
THESHOLD = 16



def main():
    """
    Select benchmarks in BENCH_DIR that cannot be solved within THESHOLD, according to the solving cached data in DICT_PATH.
    The selected benchmarks are stored in TARGET_DIR, each as a symbolic link pointing to the original benchmark file.
    """
    res_dict = {}
    solve_in_threshold = 0
    with open(DICT_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            path = row[1]
            # trimed_path = path.split("smt_lib/", 1)[1]
            solved = True if row[2] == "True" else False
            time = float(row[3])
            if solved and time <= THESHOLD: solve_in_threshold += 1
            res_dict[path] = (solved, time)
    dict_size = len(res_dict)
    print(f"not solved within {THESHOLD} sec/total size: {dict_size - solve_in_threshold}/{dict_size}")
    print(f"retain rate: {(1-solve_in_threshold/dict_size)*100:.2f}%")

    if CREATE_SET:
        bench_dir = Path(BENCH_DIR)
        target_dir = Path(TARGET_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name_counter = 0
        for file_path in bench_dir.rglob(f'*.smt2'):
            # if file_path.is_symlink():
            #     file_path = file_path.resolve()
            path_str = str(file_path)
            assert path_str in res_dict, f"{path_str} not in res_dict"
            if (not res_dict[path_str][0]) or (res_dict[path_str][1] > THESHOLD):
                target_file = target_dir / f"f{file_name_counter}.smt2"
                target_file.symlink_to(file_path)
                file_name_counter += 1
            print(f"file_name_counter" links were created!)

if __name__ == "__main__":
    main()