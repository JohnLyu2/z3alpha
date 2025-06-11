import logging
import argparse
import json
import datetime
from pathlib import Path

from z3alpha.synthesize import stage1_synthesize, parallel_linear_strategies
from z3alpha.utils import check_z3_version

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_config", type=str, help="The experiment configuration file in json"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    config = json.load(open(args.json_config, "r"))
    log_folder = Path(
        f"experiments/synthesis/out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}/"
    )
    assert not log_folder.exists()
    log_folder.mkdir(parents=True)

    z3path = config["z3path"] if "z3path" in config else "z3"
    check_z3_version(z3path)

    selected_strats, s1_time = stage1_synthesize(config, logger, log_folder)
    parallel_strat = parallel_linear_strategies(selected_strats)
    # write parallel strategy to file
    parallel_strat_path = log_folder / "parallel_strategy.txt"
    with open(parallel_strat_path, "w") as f:
        f.write(parallel_strat)
    logger.info(f"Parallel strategy saved to {parallel_strat_path}")
    # log report time
    logger.info(f"Synthesis time: {s1_time:.0f} seconds")
    

if __name__ == "__main__":
    main()
