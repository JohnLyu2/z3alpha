#!/usr/bin/env python3

import os
import sys
import glob
import random
import time  
import csv
import argparse
import json


# Parse config argument before importing machsmt
def parse_config_argument():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', 
                    type=str, 
                    required=True,
                    help='Path to the JSON configuration file')
    args, remaining = parser.parse_known_args()
    # Update sys.argv to only include remaining args for machsmt
    sys.argv[1:] = remaining
    return args.config

# Parse config argument first
config_path = parse_config_argument()
# Add the parent directory to sys.path so we can import from the sibling folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machsmt import MachSMT, Benchmark
from machsmt import args as machsmt_args
from z3alpha.evaluator import SolverRunner

def calculate_metrics(results, timeout):
    """
    Calculate PAR2 score and solving rate from results.
    
    Args:
        results: List of (solved, runtime, result) tuples
        timeout: Timeout value in seconds
    
    Returns:
        tuple: (par2_score, solving_rate)
    """
    if not results:
        return 0.0, 0.0
        
    num_solved = sum(1 for solved, _, _ in results if solved)
    total_instances = len(results)
    
    # Calculate PAR2 score
    par2_times = []
    for solved, runtime, _ in results:
        if solved:
            par2_times.append(runtime)
        else:
            par2_times.append(2 * timeout)  # PAR2 penalty
    
    avg_par2_score = sum(par2_times) / len(par2_times)
    solving_rate = (num_solved / total_instances) * 100 if total_instances > 0 else 0
    
    return avg_par2_score, solving_rate

def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def evaluate_benchmarks_in_parallel(
    solver_path: str,
    benchmarks_with_strategies_and_indices: list,  # List of (benchmark_path, strategy, original_index) tuples
    timeout: int = 300,
    tmp_dir: str = "/tmp/",
    batch_size: int = 4
):
    """
    Evaluates multiple benchmarks in parallel, each with its own strategy.
    
    Args:
        solver_path: Path to the Z3 solver
        benchmarks_with_strategies_and_indices: List of (benchmark_path, strategy, original_index) tuples
        timeout: Timeout in seconds
        tmp_dir: Directory for temporary files
        batch_size: Number of benchmarks to evaluate in parallel
        
    Returns:
        Dictionary mapping original indices to (solved, runtime, result) tuples
    """
    results = {}
    
    # Process benchmarks in batches
    for i in range(0, len(benchmarks_with_strategies_and_indices), batch_size):
        # Get current batch
        batch_end = min(i + batch_size, len(benchmarks_with_strategies_and_indices))
        current_batch = benchmarks_with_strategies_and_indices[i:batch_end]
        threads = []
        
        print(f"Processing batch {i//batch_size + 1}/{(len(benchmarks_with_strategies_and_indices) + batch_size - 1)//batch_size}")
        
        # Start a thread for each benchmark-strategy pair in the batch
        for batch_idx, (benchmark_path, strategy, original_idx) in enumerate(current_batch):
            runnerThread = SolverRunner(
                solver_path, benchmark_path, timeout, original_idx, strategy, tmp_dir
            )
            runnerThread.start()
            threads.append((runnerThread, original_idx))
        
        # Wait for all threads in this batch to complete
        time_start = time.time()
        for thread, original_idx in threads:
            time_left = max(0, timeout - (time.time() - time_start))
            thread.join(time_left)
            idx, resTask, timeTask, pathTask = thread.collect()
            solved = True if (resTask == "sat" or resTask == "unsat") else False
            results[original_idx] = (solved, timeTask, resTask)
    
    return results

def load_strategy_mapping(mapping_file):
    """Load the strategy mapping from CSV file."""
    strategies = {}
    with open(mapping_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategies[row['solver']] = row['strategy']
    return strategies

def parse_test_results(test_res_file):
    """
    Parse a CSV file containing benchmark results into a dictionary structure.
    
    Args:
        test_res_file (str): Path to the CSV file containing benchmark results
        
    Returns:
        dict: A nested dictionary with benchmark paths as outer keys, solver names as inner keys,
              and scores as values
    """
    if test_res_file is None or not os.path.exists(test_res_file):
        print(f"Error: Results file not found at {test_res_file}")
        return {}
    
    results = {}
    
    try:
        with open(test_res_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                benchmark = row['benchmark']
                solver = row['solver']
                score = float(row['score'])
                
                # Create nested dictionary structure
                if benchmark not in results:
                    results[benchmark] = {}
                
                results[benchmark][solver] = score
    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        return {}
    
    return results

def get_score(results, benchmark, solver):
    """
    Retrieve the score for a given benchmark and solver.
    
    Args:
        results (dict): The nested dictionary structure of results
        benchmark (str): The benchmark path
        solver (str): The solver name
        
    Returns:
        float or None: The score if found, None otherwise
    """
    if benchmark in results and solver in results[benchmark]:
        return results[benchmark][solver]
    return None

def main():
    # Load configuration from specified file
    config = load_config(config_path)
    
    # Get configuration values with defaults
    z3_path = config.get('paths', {}).get('z3', "z3")
    timeout = int(config.get('settings', {}).get('timeout', 300))
    tmp_dir = config.get('paths', {}).get('tmp_dir', '/tmp/')
    num_benchmarks = int(config.get('settings', {}).get('num_benchmarks', -1))
    batch_size = int(config.get('settings', {}).get('batch_size', 4))

    training_file = config.get('paths', {}).get('training_file')
    test_res_file = config.get('paths', {}).get('test_res_file')
    test_results = parse_test_results(test_res_file)

    mapping_file = config.get('paths', {}).get('mapping_file')
    output_dir = config.get('paths', {}).get('output_dir')
    default_strat_file = config.get('paths', {}).get('default_strat')

    # Get list of benchmark files
    benchmark_path = config.get('paths', {}).get('benchmark_dir')

    # Create timestamped directory for outputs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(timestamped_output_dir, exist_ok=True)
    output_file = timestamped_output_dir + "/output.txt"
    machsmt_args.results = timestamped_output_dir

    print(f"Current working directory: {os.getcwd()}")
    print(f"Training file: {training_file}")
    print(f"Using Z3 path: {z3_path}")
    print(f"Test result file: {test_res_file}")
    print(f"Timeout: {timeout} seconds")
    print(f"Using batch size of {batch_size} for parallel evaluation")
    
    # Load strategy mapping
    strategy_mapping = load_strategy_mapping(mapping_file)
    
    # Clear/create output file
    with open(output_file, 'w') as f:
        # Write the config file path to the output file
        f.write(f"Configuration file path: {os.path.abspath(config_path)}\n\n")
        
        # Write all information from the configuration file
        f.write("Configuration file contents:\n")
        f.write(json.dumps(config, indent=4))
        f.write("\n\n")
        f.write("---\n")

    # 1) Build MachSMT model with training data
    print("Building MachSMT model with Z3Alpha training data...")
    machsmt_args.logic_filter = True  # Enable logic filtering

    machsmt_args.ml_core = config.get('settings', {}).get('ml_core', "scikit")
    machsmt_args.use_gpu = config.get('settings', {}).get('use_gpu', False)
    machsmt_selector = config.get('settings', {}).get('selector', "EHM")
    run_z3alpha_strat = config.get('settings', {}).get('run_z3alpha_strat', False)

    if (machsmt_selector == "EHMLogic"):
        machsmt_args.logic_filter = True
    elif ("PWC" in machsmt_selector): 
        machsmt_args.pwc = True 
        if (machsmt_selector == "PWCLogic"): 
            machsmt_args.logic_filter = True
    else: 
        machsmt_selector = "EHM"
    
    print("Selector: ", machsmt_selector)

    model_training_start = time.time()

    machsmt = MachSMT([training_file])
    machsmt.save("lib")

    model_training_end = time.time()
    model_training_time = model_training_end - model_training_start

    print(f"Model training completed in {model_training_time:.2f} seconds")

    # 2) Evaluate using k-fold cross validation
    #print("Evaluating model using k-fold cross validation...")
    #machsmt.eval("/home/jchen688/projects/def-vganesh/jchen688/z3alpha/MachSMT/JHC/training_data/LassoRanker/test/evaluation_results_20250307_031302/benchmark_results.csv")
    
    # 3) Test on sample benchmarks
    print("Testing model on sample benchmarks...")
    
    #benchmark_path = os.path.join(root_dir, benchmark_subpath)
    print(f"Looking for benchmarks in: {benchmark_path}")
    
    benchmark_files = glob.glob(os.path.join(benchmark_path, "**/*.smt2"), recursive=True)
    if not benchmark_files:
        print(f"Warning: No benchmark files found in {benchmark_path}")
        return
        
    print(f"Found {len(benchmark_files)} benchmark files")
    
    # Randomly select benchmarks
    if (num_benchmarks<0 or num_benchmarks > len(benchmark_files)): 
        num_benchmarks = len(benchmark_files)
    selected_benchmarks = random.sample(benchmark_files, num_benchmarks)
    
    # Load model for predictions
    prediction_model = MachSMT.load("lib", with_db=False)
    
    # Load default strategy
    with open(default_strat_file, 'r') as f:
        default_strategy = f.read().strip()

    if len(default_strategy) == 0: 
        default_strategy = None
    
    # Lists to store benchmark information and results
    benchmark_info = []  # (benchmark_path, selected_solver, machsmt_strategy)
    machsmt_results = []  # (solved, runtime, result) for each benchmark
    default_results = []  # (solved, runtime, result) for each benchmark
    
    # Lists for benchmarks that need evaluation
    machsmt_evaluations = []  # (benchmark_path, strategy, index)
    default_evaluations = []  # (benchmark_path, strategy, index)

    # Process each benchmark
    for i, benchmark_path in enumerate(selected_benchmarks):
        print(f"Processing {i+1}/{len(selected_benchmarks)}: {benchmark_path}")
        
        try:
            # Create and parse benchmark
            benchmark = Benchmark(benchmark_path)
            benchmark.parse()
            
            # Get MachSMT predictions
            predictions, scores = prediction_model.predict([benchmark], include_predictions=True, selector=machsmt_selector)
            selected_solver = predictions[0].get_name()
            
            # Get the corresponding strategy
            machsmt_strategy = strategy_mapping.get(selected_solver, "Strategy not found")
            
            if machsmt_strategy == "Strategy not found":
                print(f"Warning: No strategy found for solver {selected_solver}")
                continue
            
            # Store benchmark info
            benchmark_info.append((benchmark_path, selected_solver, machsmt_strategy))
            
            # Check if we have cached results
            score = get_score(test_results, benchmark_path, selected_solver)
            if score:
                # Use cached results
                runtime_machsmt = min(timeout, score)
                solved_machsmt = False if runtime_machsmt >= timeout else True
                machsmt_results.append((solved_machsmt, runtime_machsmt, "From cache"))
            else:
                # Need to evaluate
                machsmt_evaluations.append((benchmark_path, machsmt_strategy, len(machsmt_results)))
                machsmt_results.append(None)  # Placeholder to be filled later
            
            # If we're also running default strategy
            if run_z3alpha_strat:
                default_evaluations.append((benchmark_path, default_strategy, len(default_results)))
                default_results.append(None)  # Placeholder
            else:
                default_results.append((False, 0.00, "Not executed"))
                
        except Exception as e:
            print(f"Error processing benchmark {benchmark_path}: {str(e)}")
            continue
    
    # Run MachSMT evaluations in parallel
    if machsmt_evaluations:
        print(f"Running {len(machsmt_evaluations)} MachSMT evaluations in parallel...")
        machsmt_eval_results = evaluate_benchmarks_in_parallel(
            solver_path=z3_path,
            benchmarks_with_strategies_and_indices=machsmt_evaluations,
            timeout=timeout,
            tmp_dir=tmp_dir,
            batch_size=batch_size
        )
        
        # Update the results with the evaluation results
        for idx, result in machsmt_eval_results.items():
            machsmt_results[idx] = result
    
    # Run default strategy evaluations in parallel
    if run_z3alpha_strat and default_evaluations:
        print(f"Running {len(default_evaluations)} default strategy evaluations in parallel...")
        default_eval_results = evaluate_benchmarks_in_parallel(
            solver_path=z3_path,
            benchmarks_with_strategies_and_indices=default_evaluations,
            timeout=timeout,
            tmp_dir=tmp_dir,
            batch_size=batch_size
        )
        
        # Update the results with the evaluation results
        for idx, result in default_eval_results.items():
            default_results[idx] = result
    
    # Verify all results are present
    assert all(result is not None for result in machsmt_results)
    assert all(result is not None for result in default_results)
    
    # Write results to output file
    for i, ((benchmark_path, selected_solver, machsmt_strategy), machsmt_result, default_result) in enumerate(
        zip(benchmark_info, machsmt_results, default_results), 1
    ):
        solved_machsmt, runtime_machsmt, result_machsmt = machsmt_result
        solved_default, runtime_default, result_default = default_result
        
        with open(output_file, 'a') as f:
            f.write(f"\n{i}/{len(benchmark_info)} Processing {benchmark_path}: ")
            f.write(f"Benchmark: {benchmark_path}\n")
            f.write("\nMachSMT Strategy:\n")
            f.write(f"Selected solver: {selected_solver}\n")
            f.write(f"Strategy: {machsmt_strategy}\n")
            f.write(f"Solved: {solved_machsmt}\n")
            f.write(f"Runtime: {runtime_machsmt:.2f}s\n")
            f.write(f"Result: {result_machsmt}\n")
            f.write("\nDefault Strategy:\n")
            f.write(f"Strategy: {default_strategy}\n")
            f.write(f"Solved: {solved_default}\n")
            f.write(f"Runtime: {runtime_default:.2f}s\n")
            f.write(f"Result: {result_default}\n")
            f.write("---\n")
    
    # Calculate metrics
    par2_machsmt, solving_rate_machsmt = calculate_metrics(machsmt_results, timeout)
    if run_z3alpha_strat:
        par2_default, solving_rate_default = calculate_metrics(default_results, timeout)
    else:
        par2_default, solving_rate_default = 0.00, 0.00

    print("\nSummary of results:")
    print(f"Model Training Time: {model_training_time:.2f}s")

    print("\nOverall Performance Metrics:")
    print("\nMachSMT Strategy:")
    print(f"Average PAR2 Score: {par2_machsmt:.2f}")
    print(f"Solving Rate: {solving_rate_machsmt:.2f}%")
    
    print("\nDefault Strategy:")
    print(f"Average PAR2 Score: {par2_default:.2f}")
    print(f"Solving Rate: {solving_rate_default:.2f}%")
    
    # Save metrics to file
    with open(output_file, 'a') as f:
        f.write(f"Model Training Time: {model_training_time:.2f}s\n")
        f.write("\nOverall Performance Metrics:\n")
        f.write("\nMachSMT Strategy:\n")
        f.write(f"Average PAR2 Score: {par2_machsmt:.2f}\n")
        f.write(f"Solving Rate: {solving_rate_machsmt:.2f}%\n")
        f.write("\nDefault Strategy:\n")
        f.write(f"Average PAR2 Score: {par2_default:.2f}\n")
        f.write(f"Solving Rate: {solving_rate_default:.2f}%\n")
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    sys.setrecursionlimit(10000)  # Increase from default 1000, ges stack overflow for sage2 bench_319.sh2 otherwise
    main()
