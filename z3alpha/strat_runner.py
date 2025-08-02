#!/usr/bin/env python3

import sys
import argparse
import subprocess
import os
import time
from pathlib import Path

def _get_args(argv):
    parser = argparse.ArgumentParser(description="Execute Z3 with a given strategy on a given SMT file.")
    parser.add_argument("smt_file", type=str, help="The path to the SMT file")
    parser.add_argument("--z3-path", type=str, default="z3", help="The path to the Z3 executable [default: z3]")
    parser.add_argument("--strat-file", type=str, default=None, help="The path to the strategy file (takes precedence over --strategy if both are specified)")
    parser.add_argument("--strategy", type=str, default=None, help="The strategy string to use (if both --strategy and --strat-file are specified, --strat-file takes precedence)")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/", help="The path to the temporary directory [default: /tmp/]")
    return parser.parse_args(argv)

def rewrite_smt2_with_strat(smt2_str, strat):
    new_smt2_str = ""
    for line in smt2_str.split('\n'):
        if "check-sat" in line:
            new_smt2_str += f"(check-sat-using {strat})\n"
        else:
            new_smt2_str += line + "\n"
    return new_smt2_str

def main(argv=sys.argv[1:]):
    """
    Execute Z3 with a given strategy on a given SMT file.
    """
    args = _get_args(argv)
    z3_path = args.z3_path
    smt_file = args.smt_file
    strat_file = args.strat_file
    strategy = args.strategy
    tmp_dir = args.tmp_dir

    assert Path(tmp_dir).exists(), f"The tmp directory {tmp_dir} does not exist"
    assert Path(smt_file).exists(), f"The SMT file {smt_file} does not exist"
    assert Path(z3_path).exists(), f"The Z3 executable {z3_path} does not exist"

    if strat_file:
        with open(strat_file, "r") as f:
            strat = f.read()
    else:
        strat = strategy
    if strat is not None:
        smt2_str = Path(smt_file).read_text()
        new_smt2_str = rewrite_smt2_with_strat(smt2_str, strat)
        new_smt2_file = Path(tmp_dir) / f"tmp_{os.getpid()}_{int(time.time() * 1000)}.smt2"
        new_smt2_file.write_text(new_smt2_str)
        smt_file = new_smt2_file

    try:
        result = subprocess.run([z3_path, smt_file], capture_output=True, text=True, check=True)
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
    except subprocess.CalledProcessError as e:
        if e.stdout:
            sys.stdout.write(e.stdout)
        if e.stderr:
            sys.stderr.write(e.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    finally:
        # delete the temporary file if it exists
        try:
            new_smt2_file.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    main()