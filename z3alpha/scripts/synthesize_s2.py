import logging
import argparse
import json
import csv
import datetime
import os

from ..synthesize import cache4stage2, stage2_synthesize
from ..utils import read_strat_res_from_csv

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")
)
log.addHandler(log_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_config", type=str, help="The experiment configuration file in json"
    )
    parser.add_argument(
        "--no_caching",
        action="store_false",
        default=True,
        help="cache results are provided; no need to caching s1 strats",
    )
    parser.add_argument(
        "--benchlst4cahce",
        action="store_true",
        default=False,
        help="bench list is provided for caching",
    )  # this argument states you can read a benchlist from ln_res and cache for the benchmarks in this list; don't remember when it will be useful
    parser.add_argument(
        "--no_syn",
        action="store_false",
        default=True,
        help="does not need to perform s2 synthesize",
    )
    configJsonPath = parser.parse_args()
    config = json.load(open(configJsonPath.json_config, "r"))
    # num_strat = config['ln_strat_num']
    s2config = config["s2config"]
    is_cache = configJsonPath.no_caching
    is_cache_benchlst = configJsonPath.benchlst4cahce
    is_syn = configJsonPath.no_syn
    assert is_cache or is_syn, "no caching and no synthesize; nothing to do"
    log_folder = (
        f"experiments/synthesis/out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}/"
    )
    assert not os.path.exists(log_folder)
    os.makedirs(log_folder)
    ln_strat_file = s2config["ln_res"]
    res_lst, bench_lst = read_strat_res_from_csv(ln_strat_file)
    assert len(res_lst) > 0

    cache_time = 0
    if is_cache:
        # res_lst is a list of (strat, res_lst)
        stratLst = [x[0] for x in res_lst]
        if is_cache_benchlst:
            assert len(bench_lst) > 0
        else:
            bench_lst = None
        res_lst, bench_lst, cache_time = cache4stage2(
            stratLst, config, log, log_folder, benchlst=bench_lst
        )

    if not is_syn:
        return

    if not is_cache:
        assert len(bench_lst) > 0
        log.info(f"s2syn: read {len(res_lst)} strategies from {ln_strat_file}")

    bst_strats, s2mcts_time = stage2_synthesize(
        res_lst, bench_lst, config, log, log_folder
    )
    order = 1
    for bst_strat in bst_strats:
        log.info(f"Best strategy {order}: {bst_strat}")
        order += 1
    log.info(
        f"S2: Cache time: {cache_time:.0f}, MCTS time: {s2mcts_time:.0f}, Total time: {cache_time + s2mcts_time:.0f}"
    )


if __name__ == "__main__":
    main()
