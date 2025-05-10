import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import sys
import os

logic_strategy = {"QF_SLIA": "/home/paul/Dokumente/z3alpha/test.txt"}


def read_smtlib_logic(smt_file):
    with open(smt_file, "r") as f:
        for line in f:
            if line.startswith("(set-logic"):
                return line.split()[1][:-1]
    return None


def run_z3_with_command(smt_file, command, logic):
    start_time = time.time()
    content = ""
    with open(smt_file, "r") as f:
        for line in f:
            if "check-sat" in line:
                content += f"(check-sat-using {command})\n"
            else:
                content += line + "\n"

    temp_file = f"temp_{hash(command)}.smt2"
    with open(temp_file, "w") as f:
        f.write(content)
    if logic in ["QF_S", "QF_SLIA", "QF_SNIA"]:
        result = subprocess.run(
            ["z3", "z3str", temp_file], capture_output=True, text=True
        )
    else:
        result = subprocess.run(["z3", temp_file], capture_output=True, text=True)

    elapsed_time = time.time() - start_time

    return {
        "command": command,
        "output": result.stdout.strip(),
        "execution_time": elapsed_time,
    }


def main(smt_file):
    results = []
    logic = read_smtlib_logic(smt_file)
    print(logic)
    command_file = logic_strategy[logic]
    with open(command_file, "r") as f:
        commands = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_z3_with_command, smt_file, command, logic)
            for command in commands
        ]
        for future in futures:
            results.append(future.result())

    with open("/home/paul/Dokumente/z3alpha/alphasmt/output.txt", "w") as f:
        for idx, result in enumerate(results):
            f.write(f"Command {idx + 1}:\n")
            f.write(f"{result['command']}\n\n")
            f.write("Output:\n")
            f.write(f"{result['output']}\n\n")
            f.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
            f.write("=" * 50 + "\n\n")


if __name__ == "__main__":
    main("/home/paul/Dokumente/z3alpha/smtcomp24/test_instances/qfslia0.smt2")
