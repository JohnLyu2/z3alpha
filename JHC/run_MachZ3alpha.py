import logging
import argparse
import json
import datetime
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from z3alpha.synthesize import stage1_synthesize, stage2_synthesize, cache4stage2

# Set up logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")
)
log.addHandler(log_handler)


def run_stage1(config):
    """Run stage 1 and return the path to the generated output folder"""
    # Get output directory from config or use default
    output_base_dir = config.get("output_dir", "experiments/synthesis")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Create timestamped folder within the specified output directory
    log_folder = os.path.join(output_base_dir, f"out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}/")
    assert not os.path.exists(log_folder)
    os.makedirs(log_folder)
    
    # Run stage 1 synthesis
    log.info("Starting Stage 1 synthesis...")
    selected_strats, s1_time = stage1_synthesize(config, log, log_folder)
    
    # Cache data for stage 2
    res_dict, bench_lst, cache_time = cache4stage2(
        selected_strats, config, log, log_folder
    )
    
    # Run stage 2 synthesis
    bst_strats, s2mcts_time = stage2_synthesize(
        res_dict, bench_lst, config, log, log_folder
    )
    
    # Log results
    order = 1
    for bst_strat in bst_strats:
        log.info(f"Best strategy {order}: {bst_strat}")
        order += 1
    log.info(
        f"S1 time: {s1_time:.0f}, S2 Cache time: {cache_time:.0f}, S2 MCTS time: {s2mcts_time:.0f}, Total time: {s1_time + cache_time + s2mcts_time:.0f}"
    )
    
    # Return the path to the generated folder
    return {
        "log_folder": os.path.abspath(log_folder),
        "times": {
            "s1_time": s1_time,
            "cache_time": cache_time,
            "s2mcts_time": s2mcts_time,
            "total_time": s1_time + cache_time + s2mcts_time
        }
    }


def prepare_stage2_config(machsmt_config_path, stage1_results):
    """Update the stage 2 config with paths from stage 1 results"""
    with open(machsmt_config_path, 'r') as f:
        config = json.load(f)
    
    # Get the absolute path to the stage 1 output folder
    stage1_folder = stage1_results["log_folder"]
    
    # Update the paths in the stage 2 config
    config["paths"]["training_file"] = os.path.join(stage1_folder, "ln_res.csv")
    config["paths"]["mapping_file"] = os.path.join(stage1_folder, "strat_mapping.csv")
    config["paths"]["default_strat"] = os.path.join(stage1_folder, "final_strategy1.txt")
    
    # Create a new config file with the updated paths
    new_config_path = os.path.join(stage1_folder, "stage2_config.json")
    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    log.info(f"Created stage 2 config file at: {new_config_path}")
    return new_config_path


def run_stage2(stage1_results, machsmt_config_path):
    """Run stage 2 using the updated config"""
    # Prepare stage 2 config file
    stage2_config_path = prepare_stage2_config(machsmt_config_path, stage1_results)
    
    # Path to the stage 2 script
    stage2_script = "JHC/run_MachSMT.py"
    
    log.info(f"Starting Stage 2 with config: {stage2_config_path}")
    
    try:
        # Run stage 2 script with the updated config
        cmd = [sys.executable, stage2_script, "--config", stage2_config_path]
        log.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("Stage 2 completed successfully")
        log.info(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Stage 2 failed with exit code {e.returncode}")
        log.error(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=str, help="The experiment configuration file in json"
    )

    parser.add_argument(
        "--skip-stage2", action="store_true", help="Skip running stage 2"
    )
    
    args = parser.parse_args()
    
    # Load the configuration
    config_path = args.config
    if not os.path.exists(config_path):
        log.error(f"Config file not found: {config_path}")
        return
    
    config = json.load(open(config_path, "r"))
    
    # Check for machsmt_config
    machsmt_config_path = config.get("machsmt_config")
    if not args.skip_stage2 and (not machsmt_config_path or not os.path.exists(machsmt_config_path)):
        log.error(f"MachSMT config file not found or not specified: {machsmt_config_path}")
        return
    
    # Run stage 1
    log.info("Starting stage 1...")
    stage1_results = run_stage1(config)
    log.info(f"Stage 1 completed. Output folder: {stage1_results['log_folder']}")
    
    # Run stage 2 if not skipped
    if not args.skip_stage2:
        # Run stage 2
        success = run_stage2(stage1_results, machsmt_config_path)
        if success:
            log.info("Full workflow completed successfully!")
        else:
            log.error("Workflow failed during stage 2")
    else:
        log.info("Stage 2 skipped. Workflow completed.")


if __name__ == "__main__":
    main()