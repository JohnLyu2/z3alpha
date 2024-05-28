import sys
import os
import subprocess
from strat_dict import logic_strategy

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

def main():
    if len(sys.argv) != 2:
        print("Usage: python z3alpha.py <path_to_smt2_file>")
        return
    
    smt2_path = sys.argv[1]
    with open(smt2_path, 'r') as f:
        smt2_str = f.read()

    solver_path = "./z3bin/z3"

    logic = read_smtlib_logic(smt2_str)
    if logic and (logic in logic_strategy):
        strat_path = logic_strategy[logic]
        # check whether strat_path exists
        if os.path.exists(strat_path):
            with open(strat_path, 'r') as f:
                strat = f.read()
            smt2_str = rewrite_smt2_with_strat(smt2_str, strat)

    # write into a new smt2 file in tmp
    new_smt2_path = "/tmp/rw_instance.smt2"
    with open(new_smt2_path, 'w') as f:
        f.write(smt2_str)

    # run z3 with the new smt2 file
    cmd = f"{solver_path} {new_smt2_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Print the standard output and standard error
    print(result.stdout)

if __name__ == "__main__":
    main()