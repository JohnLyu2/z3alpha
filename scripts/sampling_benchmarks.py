import csv
from pathlib import Path
import random
import argparse

THRESHOLD = 1  

def parse_args():
    parser = argparse.ArgumentParser(description='Select and create symbolic links for benchmark files based on solving time.')
    parser.add_argument('--performance-path', required=True, help='Path to the CSV file containing solving results and times')
    parser.add_argument('--target-dir', required=True, help='Directory where symbolic links will be created')
    parser.add_argument('--target-size', type=int, required=True, help='Number of benchmarks to select')
    return parser.parse_args()

def main():
    """
    Select benchmarks in BENCH_DIR that cannot be solved within THRESHOLD seconds.
    If the number of such benchmarks exceeds TARGET_SIZE, perform weighted random selection:
    - Weight 1 for solved benchmarks
    - Weight 2 for unsolved benchmarks
    The selected benchmarks are stored in TARGET_DIR as symbolic links.
    """
    args = parse_args()
    
    res_dict = {}
    slow_benchmarks = []  # List to store benchmarks that take >THRESHOLD second
    
    with open(args.performance_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            path = row[1]
            solved = True if row[2] == "True" else False
            time = float(row[3])
            res_dict[path] = (solved, time)
            if time > THRESHOLD:
                slow_benchmarks.append((path, solved))
    
    print(f"Found {len(slow_benchmarks)} benchmarks taking more than {THRESHOLD} second")
    
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # If we have more benchmarks than target size, perform weighted selection
    if len(slow_benchmarks) > args.target_size:
        # Create weights: 1 for solved, 2 for unsolved
        weights = [2 if not solved else 1 for _, solved in slow_benchmarks]
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Select benchmarks using weighted random choice
        selected_indices = random.choices(
            range(len(slow_benchmarks)), 
            weights=weights, 
            k=args.target_size
        )
        selected_benchmarks = [slow_benchmarks[i] for i in selected_indices]
    else:
        selected_benchmarks = slow_benchmarks
    
    # Create symbolic links for selected benchmarks
    file_name_counter = 0
    for path, solved in selected_benchmarks:
        file_path = Path(path)
        target_file = target_dir / f"f{file_name_counter}.smt2"
        target_file.symlink_to(file_path)
        file_name_counter += 1
        
    print(f"Created {file_name_counter} symbolic links at {target_dir.absolute()}")
    print(f"Selected {sum(1 for _, solved in selected_benchmarks if not solved)} unsolved and "
          f"{sum(1 for _, solved in selected_benchmarks if solved)} solved benchmarks")

if __name__ == "__main__":
    main()