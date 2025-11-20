#!/usr/bin/env python3

import os
import sys
import glob
import random
import time
import csv
import argparse
import json
import datetime

# Parse config argument
def parse_config_argument():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    parser.add_argument('--use_description', action='store_true', help='Enable description features in MachSMT')
    parser.add_argument('--use_family', action='store_true', help='Enable family features in MachSMT')
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    return args

args = parse_config_argument()
config_path = args.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machsmt import MachSMT, Benchmark
from machsmt import args as machsmt_args

# Load configuration
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# Convert full path to relative benchmark path (from 'benchmarks/')
def relative_benchmark_path(full_path):
    parts = full_path.split("benchmarks/")
    if len(parts) > 1:
        return parts[1]
    else:
        # fallback: use basename if split fails
        return os.path.basename(full_path)

# Parse CSV test results
def parse_test_results(test_res_file):
    results = {}
    if test_res_file is None or not os.path.exists(test_res_file):
        print(f"[DEBUG] Error: Results file not found at {test_res_file}")
        return results

    print(f"[DEBUG] Loading test results from {test_res_file}")
    with open(test_res_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            benchmark = relative_benchmark_path(row['benchmark'])
            solver = row['solver']
            score = float(row['score'])
            if benchmark not in results:
                results[benchmark] = {}
            results[benchmark][solver] = score
    print(f"[DEBUG] Loaded results for {len(results)} benchmarks")
    return results

def get_score(results, benchmark, solver):
    benchmark_rel = relative_benchmark_path(benchmark)
    if benchmark_rel in results and solver in results[benchmark_rel]:
        return results[benchmark_rel][solver]
    return None

def main():
    # Load configuration
    config = load_config(config_path)

    # Paths
    training_file = config["paths"]["training_file"]
    test_res_file = config["paths"]["test_res_file"]
    benchmark_dir = config["paths"]["benchmark_dir"]
    output_dir = config["paths"]["output_dir"]

    # Settings
    timeout = int(config["settings"].get("timeout", 300))
    num_benchmarks = int(config["settings"].get("num_benchmarks", -1))
    ml_core = config["settings"].get("ml_core", "scikit")
    use_gpu = config["settings"].get("use_gpu", False)
    selector = config["settings"].get("selector", "EHM")
    description = config["settings"].get("description", False)

    # Parse precomputed test results
    test_results = parse_test_results(test_res_file)

    # Collect benchmark files
    benchmark_files = glob.glob(os.path.join(benchmark_dir, "**/*.smt2"), recursive=True)
    print(f"[DEBUG] Found {len(benchmark_files)} benchmark files in {benchmark_dir}")

    if not benchmark_files:
        print(f"[DEBUG] No benchmarks found in {benchmark_dir}")
        return

    if num_benchmarks < 0 or num_benchmarks > len(benchmark_files):
        num_benchmarks = len(benchmark_files)
    selected_benchmarks = random.sample(benchmark_files, num_benchmarks)
    print(f"[DEBUG] Selected {len(selected_benchmarks)} benchmarks for processing")

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamped_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(timestamped_output_dir, exist_ok=True)
    output_file = os.path.join(timestamped_output_dir, "output.txt")
    machsmt_args.results = timestamped_output_dir

    # Set MachSMT arguments from command-line flags
    machsmt_args.description = args.use_description
    machsmt_args.description_family = args.use_family

    # print which one we're using 
    if machsmt_args.description: 
        print("Using description embeddings")
    elif machsmt_args.description_family: 
        print("Using family")
    else: 
        print("Using default features space")

    # Train MachSMT model
    print("[DEBUG] Training MachSMT model...")
    model_training_start = time.time()
    machsmt = MachSMT([training_file])
    machsmt.save("lib")
    model_training_end = time.time()
    model_training_time = model_training_end - model_training_start
    print(f"[DEBUG] Model trained in {model_training_time:.2f} seconds")

    # Load trained model
    prediction_model = machsmt #MachSMT.load("lib", with_db=False)

    # Process each benchmark
    with open(output_file, 'w') as f:
        f.write(f"Configuration file: {os.path.abspath(config_path)}\n\n")
        f.write("Configuration contents:\n")
        f.write(json.dumps(config, indent=4))
        f.write("\n\n---\n")

    benchmark_info = []
    machsmt_results = []

    for i, benchmark_path in enumerate(selected_benchmarks, 1):
        print(f"[DEBUG] Processing {i}/{len(selected_benchmarks)}: {benchmark_path}")

        try:
            benchmark = Benchmark(benchmark_path)
            benchmark.parse()

            # Get MachSMT predictions
            predictions, scores = prediction_model.predict([benchmark], include_predictions=True, selector=selector)
            selected_solver = predictions[0].get_name()
            print(f"[DEBUG] Predicted solver: {selected_solver} for benchmark {benchmark_path}")

            # Retrieve precomputed result from CSV
            score = get_score(test_results, benchmark_path, selected_solver)
            if score is None:
                print(f"[DEBUG] No precomputed result for benchmark {benchmark_path} with solver {selected_solver}")
                continue

            # Store results
            runtime = min(timeout, score)
            solved = runtime < timeout
            machsmt_results.append((solved, runtime, "From CSV"))
            benchmark_info.append((benchmark_path, selected_solver))
            print(f"[DEBUG] Benchmark {benchmark_path}: solved={solved}, runtime={runtime:.2f}s")

            # Write to output
            with open(output_file, 'a') as f:
                f.write(f"{i}/{len(selected_benchmarks)}: Benchmark {benchmark_path}\n")
                f.write(f"Predicted solver: {selected_solver}\n")
                f.write(f"Solved: {solved}\n")
                f.write(f"Runtime: {runtime:.2f}s\n")
                f.write("---\n")

        except Exception as e:
            print(f"[DEBUG] Error processing {benchmark_path}: {e}")
            continue

    # Summary
    total_solved = sum(1 for solved, _, _ in machsmt_results if solved)
    num_results = len(machsmt_results)
    if num_results > 0:
        print(f"[DEBUG] Solved {total_solved}/{num_results} benchmarks ({100 * total_solved / num_results:.2f}%)")
    else:
        print(f"[DEBUG] No benchmarks were processed successfully. Check selector or CSV matching!")

    print(f"\n[DEBUG] Model training time: {model_training_time:.2f}s")
    print(f"[DEBUG] Results saved to {output_file}")

if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    main()
