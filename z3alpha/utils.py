import csv
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


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
