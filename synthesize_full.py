import logging 
import argparse
import json
import datetime
import os

from alphasmt.synthesize import *

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s','%Y-%m-%d %H:%M:%S'))
log.addHandler(log_handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='The experiment configuration file in json')
    configJsonPath = parser.parse_args()
    config = json.load(open(configJsonPath.json_config, 'r'))
    # num_strat = config['ln_strat_num']
    log_folder = f"experiments/synthesis/out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}/"
    assert(not os.path.exists(log_folder))
    os.makedirs(log_folder)
    selected_strats, s1_time = stage1_synthesize(config, log, log_folder)
    res_dict, bench_lst, cache_time = cache4stage2(selected_strats, config, log, log_folder)
    bst_strats, s2mcts_time = stage2_synthesize(res_dict, bench_lst, config, log, log_folder)
    order = 1
    for bst_strat in bst_strats:
        log.info(f"Best strategy {order}: {bst_strat}")
        order += 1
    log.info(f"S1 time: {s1_time:.0f}, S2 Cache time: {cache_time:.0f}, S2 MCTS time: {s2mcts_time:.0f}, Total time: {s1_time + cache_time + s2mcts_time:.0f}")

if __name__ == "__main__":
    main()