import csv
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)
import os

def solvedNum(resLst):
    return len([res for res in resLst if res[0]])


def solvedNumReward(resLst):
    return solvedNum(resLst) / len(resLst)


def parN(resLst, n, timeout):
    parN = 0
    for i in range(len(resLst)):
        if not resLst[i][0]:
            parN += n * timeout
        else:
            parN += resLst[i][1]
    return parN


def parNReward(resLst, n, timeout):
    par_n = parN(resLst, n, timeout)
    maxParN = len(resLst) * timeout * n
    return 1 - par_n / maxParN


def calculatePercentile(lst, percentile):
    assert len(lst) > 0
    # percentile is of the form like "90p"
    percent = float(percentile[:-1]) / 100
    sortedLst = sorted(lst)
    index = int(len(sortedLst) * percent)
    return sortedLst[index]


def write_strat_res_to_csv(res_lst, csv_path, bench_lst):
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(["strat"] + bench_lst)
        for strat, res in res_lst:
            write_lst = []
            for res_tuple in res:
                if res_tuple[0]:
                    write_lst.append(res_tuple[1])
                else:
                    write_lst.append(-res_tuple[1])
            writer.writerow([strat] + write_lst)


def read_strat_res_from_csv(csv_path):
    results = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        bench_lst = header[1:]
        for row in reader:
            strat = row[0]
            res_lst = []
            for res in row[1:]:
                stime = float(res)
                if stime < 0:
                    res_lst.append((False, -stime, "na"))
                else:
                    res_lst.append((True, stime, "na"))
            results.append((strat, res_lst))
    return results, bench_lst


def check_z3_version(z3path):
    try:
        result = subprocess.run([z3path, "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to get Z3 version. Error: {result.stderr}")
            sys.exit(1)
        
        version = result.stdout.strip()
        logger.info(f"Using Z3 version: {version}")
        return version
    except Exception as e:
        logger.error(f"Error checking Z3 version: {str(e)}")
        sys.exit(1)

def write_strat_res_to_csv_long_format(res_lst, csv_path, bench_lst):
    """
    Writes strategy results to CSV in a long format, differing from write_strat_res_to_csv() in several ways:
    1. Format: Writes in long format (one row per benchmark-strategy pair) vs wide format (one row per strategy)
    2. Path handling: Converts relative benchmark paths to absolute paths
    3. Strategy naming: Creates a mapping between original strategies and simplified names (Z3_strat_0, Z3_strat_1, etc.)
       and stores this mapping in a separate 'strat_mapping.csv' file in the same directory
    4. Score calculation: When res_tuple[0] is False, multiplies score by 2 instead of negating it (par2)
    
    Input format remains the same:
    - res_lst: List of tuples (strategy, results)
    - csv_path: Path where to save the main results CSV
    - bench_lst: List of benchmark paths (in relative form)
    
    Creates two files:
    1. Main CSV with columns: benchmark,solver,score
    2. strat_mapping.csv with columns: solver,strategy
    """

    # Create mapping file path in same directory as csv_path
    mapping_path = os.path.join(os.path.dirname(csv_path), "strat_mapping.csv")
    
    # Create strategy mapping
    strat_mapping = {}
    for i, (strat, _) in enumerate(res_lst):
        strat_mapping[strat] = f"Z3_strat_{i}"
    
    # Write strategy mapping to separate CSV
    with open(mapping_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["solver", "strategy"])
        for strat, solver_name in strat_mapping.items():
            writer.writerow([solver_name, strat])
    
    # Write main results
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(["benchmark", "solver", "score"])
        
        # Convert relative paths to absolute paths
        abs_bench_lst = [os.path.abspath(bench) for bench in bench_lst]
        
        # For each strategy and its results
        for strat, res in res_lst:
            solver_name = strat_mapping[strat]
            # For each benchmark and its corresponding result
            for bench, res_tuple in zip(abs_bench_lst, res):
                score = res_tuple[1] if res_tuple[0] else res_tuple[1] * 2
                writer.writerow([bench, solver_name, score])