import sys
import argparse
import logging
from .multiprocess_evaluator import run_solver

def _get_args(argv):
    parser = argparse.ArgumentParser(
        description="Z3Alpha: Execute Z3 solver with specified strategy on benchmark instances"
    )
    parser.add_argument("benchmark_path", help="Path to the benchmark/instance file")
    parser.add_argument(
        "--strategy",
        help="Strategy to use for solving (optional)",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--z3-binary-path",
        help="path to Z3 solver executable",
        type=str,
        default="z3",
        required=False,
    )
    parser.add_argument(
        "--timeout",
        help="timeout in seconds",
        type=int,
        default=300,
        required=False,
    )
    parser.add_argument(
        "--tmp-dir",
        help="temporary directory for modified SMT files",
        type=str,
        default="/tmp/",
        required=False,
    )
    parser.add_argument(
        "--debug",
        help="show debugging messages",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        help="show verbose output",
        action="store_true",
        required=False,
    )
    return parser.parse_args(argv)


def main(argv=sys.argv[1:]):
    args = _get_args(argv)
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    try:
        # Call the existing run_solver function
        task_id, result_status, runtime, smt_file = run_solver(
            solver_path=args.z3_binary_path,
            smt_file=args.benchmark_path,
            timeout=args.timeout,
            id=0,  # Single task, so ID is 0
            strategy=args.strategy,
            tmp_dir=args.tmp_dir,
            cpu_limit=0,
            memory_limit=None,
            monitor_resources=False
        )
        
        # Prepare result dictionary
        result = {
            'benchmark': smt_file,
            'strategy': args.strategy,
            'status': result_status,
            'runtime': runtime,
            'timeout': args.timeout,
        }
                
        # Log result
        if args.verbose:
            logging.info("Solving completed:")
            logging.info(f"  Status: {result_status}")
            logging.info(f"  Runtime: {runtime:.3f}s")
            logging.info(f"  Benchmark: {smt_file}")
            logging.info(f"  Strategy: {args.strategy}")
        else:
            logging.info(f"Result: {result_status} ({runtime:.3f}s)")
        
        if result_status in ['sat', 'unsat']:
            return 0  # Success
        else:
            return 1  # Error/timeout
            
    except Exception as e:
        logging.error("Error during solving: %s", str(e))
        if args.debug:
            import traceback
            logging.error("Traceback: %s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())