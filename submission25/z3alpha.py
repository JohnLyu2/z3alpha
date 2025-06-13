#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import json
import time
import os
import argparse

def load_logic_strategies():
    script_path = Path(__file__).resolve()
    z3alpha_dir = script_path.parent
    strategies_path = z3alpha_dir / "strats" / "logic_strategies.json"
    with open(strategies_path, 'r') as f:
        return json.load(f)

def read_smtlib_logic(smt2_str):
    for line in smt2_str.split('\n'):
        if line.startswith('(set-logic'):
            return line.split()[1][:-1]
    return None

def rewrite_smt2_with_strat(smt2_str, strat):
    new_smt2_str = ""
    for line in smt2_str.split('\n'):
        if "check-sat" in line:
            new_smt2_str += f"(check-sat-using {strat})\n"
        else:
            new_smt2_str += line + "\n"
    return new_smt2_str

def debug_print(msg, debug=False):
    if debug:
        print(f"[DEBUG] {msg}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Run Z3 with appropriate strategy based on SMT-LIB logic.')
    parser.add_argument('smt2_path', help='Path to the SMT2 file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    z3alpha_dir = script_path.parent
    
    smt2_path = Path(args.smt2_path)
    debug_print(f"Processing SMT2 file: {smt2_path}", args.debug)
    
    with open(smt2_path, 'r') as f:
        smt2_str = f.read()

    solver_path = z3alpha_dir / "z3bin" / "z3"
    debug_print(f"Using solver: {solver_path}", args.debug)

    logic = read_smtlib_logic(smt2_str)
    debug_print(f"Detected logic: {logic}", args.debug)

    if logic:
        logic_strategies = load_logic_strategies()
        if logic in logic_strategies:
            strat_filename = logic_strategies[logic]
            strat_path = z3alpha_dir / "strats" / strat_filename
            debug_print(f"Found strategy file: {strat_path}", args.debug)
            
            # check whether strat_path exists
            if strat_path.exists():
                with open(strat_path, 'r') as f:
                    strat = f.read()
                debug_print(f"Using strategy: {strat.strip()}", args.debug)
                smt2_str = rewrite_smt2_with_strat(smt2_str, strat)
            else:
                debug_print(f"Strategy file not found: {strat_path}", args.debug)
        else:
            debug_print(f"No strategy found for logic: {logic}", args.debug)

    # write into a new smt2 file in tmp with unique name
    timestamp = int(time.time() * 1000)  # milliseconds
    pid = os.getpid()
    rw_smt2_path = Path(f"/tmp/{pid}_{timestamp}.smt2")
    debug_print(f"Writing temporary file: {rw_smt2_path}", args.debug)
    
    with open(rw_smt2_path, 'w') as f:
        f.write(smt2_str)

    try:
        debug_print(f"Running Z3 with command: {solver_path} {rw_smt2_path}", args.debug)
        result = subprocess.run(
            [str(solver_path), str(rw_smt2_path)],
            capture_output=True,
            text=True,
            check=True
        )
        # Print stdout to stdout and stderr to stderr
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running Z3: {e}", file=sys.stderr)
        if e.stdout:
            sys.stdout.write(e.stdout)
        if e.stderr:
            sys.stderr.write(e.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up the temporary file
        try:
            rw_smt2_path.unlink()
            debug_print(f"Cleaned up temporary file: {rw_smt2_path}", args.debug)
        except Exception:
            pass  # Ignore cleanup errors

if __name__ == "__main__":
    main()