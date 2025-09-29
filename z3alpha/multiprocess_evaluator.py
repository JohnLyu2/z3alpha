import os
import multiprocessing
import subprocess
import time
import logging
import threading
from pathlib import Path
from z3alpha.resource_monitor import ResourceMonitor, log_resource_usage
from z3alpha.utils import setup_logging
from z3alpha.resource_allocation import set_cpu_affinity, calculate_resource_allocation
from z3alpha.resource_logging import log_execution_summary, log_batch_progress

log = logging.getLogger(__name__)


def build_solver_command(solver_path, smt_file, additional_args=None):
    cmd = [str(solver_path), str(smt_file)]
    
    if additional_args:
        cmd.extend(str(arg) for arg in additional_args)
    
    return cmd

def parse_strategy_file(strategy_path):
    if not strategy_path:
        return None
    
    try:
        with open(strategy_path, 'r') as f:
            strategy = f.read().strip()
        
        if not strategy:
            raise ValueError(f"Strategy file {strategy_path} is empty")
        
        log.info(f"Strategy loaded from {strategy_path}: {strategy[:100]}{'...' if len(strategy) > 100 else ''}")
        return strategy
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
    except Exception as e:
        raise Exception(f"Error reading strategy file {strategy_path}: {e}")

def run_solver(
    solver_cmd, smt_file, timeout, task_id, 
    cpu_limit=1, memory_limit=None, monitor_resources=False
):
    """Enhanced runner with resource monitoring - now using CLI entry point"""
    
    if cpu_limit > 0:
        process = psutil.Process()
        set_cpu_affinity(process, task_id, cpu_limit, process.cpu_affinity())    
        
    # Log process info
    log.debug(f"Task {task_id}: Starting solver process (PID: {os.getpid()})")
    if memory_limit:
        log.debug(f"Task {task_id}: Memory limit: {memory_limit} MB")
    
    # Start resource monitoring for this process if requested
    if monitor_resources:
        monitor_thread = threading.Thread(
            target=log_resource_usage, 
            args=(os.getpid(), task_id, min(timeout, 60)),  # Monitor for up to 60 seconds
            daemon=True
        )
        monitor_thread.start()
    
    # Build command 
    cmd = solver_cmd + [str(smt_file)]    
    log.info(f"Task {task_id}: Running command: {' '.join(cmd)}")
    
    # Run the solver
    time_before = time.time()
    
    # Build full command
    cmd = solver_cmd + [str(smt_file)]
    log.info(f"Task {task_id}: Running: {' '.join(cmd)}")
    
    # Execute solver
    time_before = time.time()
    
    try:
        if memory_limit:
            # Use ulimit for memory constraint
            ulimit_cmd = f"ulimit -v {memory_limit * 1024} && {' '.join(cmd)}"
            p = subprocess.Popen(ulimit_cmd, stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, shell=True)
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        log.debug(f"Task {task_id}: Subprocess PID: {p.pid}")
        
        # Wait for completion
        out, err = p.communicate(timeout=timeout)
        runtime = time.time() - time_before
        
        # Parse result
        lines = out.decode("utf-8").split("\n")
        result = lines[0] if lines and len(lines[0]) > 0 else "error"
        
        log.info(f"Task {task_id}: Completed in {runtime:.2f}s → {result}")
        
        # Check for errors
        if result.startswith("(error") or err:
            log.warning(f"Task {task_id}: Error - {result}, stderr: {err.decode('utf-8')}")
            return task_id, "error", runtime, smt_file
        
        return task_id, result, runtime, smt_file
    
    except subprocess.TimeoutExpired:
        log.info(f"Task {task_id}: Timeout after {timeout}s")
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        return task_id, "timeout", timeout, smt_file
        
def task_runner(args):
    """Wrapper for parallel execution"""
    smt_file, task_id, solver_cmd, timeout, cpu_limit, memory_limit, monitor_resources = args
    return run_solver(
        solver_cmd=solver_cmd,
        smt_file=smt_file,
        timeout=timeout,
        task_id=task_id,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        monitor_resources=monitor_resources
    )

class SolverEvaluator:
    def __init__(
        self,
        solver_cmd,
        benchmark_lst,
        timeout,
        cpus_per_task,
        is_write_res=False,
        res_path=None,
        memory_per_task=None,
        disable_cpu_affinity=False,
        monitor_output_dir=None
    ):
        self.solverCmd = solver_cmd
        self.benchmarkLst = benchmark_lst
        self.timeout = timeout
        self.isWriteRes = is_write_res
        self.resPath = res_path
        self.disable_cpu_affinity = disable_cpu_affinity
        
        assert self.getBenchmarkSize() > 0
        assert self.timeout > 0

        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(monitor_output_dir)
        
        # Calculate optimal resource allocation
        allocation = calculate_resource_allocation(
            cpus_per_task=cpus_per_task,
            memory_per_task=memory_per_task,
            num_benchmarks=self.getBenchmarkSize()
        )
        
        # Validate and log allocation
        allocation.validate()
        allocation.log_summary(len(benchmark_lst))
        
        # Store allocation results
        self.batchSize = allocation.batch_size
        self.maxParallelTasks = allocation.batch_size
        self.cpusPerTask = allocation.cpus_per_task
        self.memoryPerTask = allocation.memory_per_task

    def getBenchmarkSize(self):
        return len(self.benchmarkLst)

    def getResLst(self):
        size = self.getBenchmarkSize()
        results = [None] * size
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        log_execution_summary(
            num_benchmarks=size,
            batch_size=self.batchSize,
            cpus_per_task=self.cpusPerTask,
            memory_per_task=self.memoryPerTask
        )       
         
        pool = multiprocessing.Pool(processes=self.maxParallelTasks)
        
        try:
            for i in range(0, size, self.batchSize):
                batch_end = min(i + self.batchSize, size)
                
                batch_num = i // self.batchSize + 1
                batch_start_time = time.time()
                log_batch_progress(batch_num, i, batch_end)  
                              
                batch_args = [
                    (
                        self.benchmarkLst[idx],  # smt_file
                        idx,                      # task_id
                        self.solverCmd,          # solver_cmd (list)
                        self.timeout,            # timeout
                        0 if self.disable_cpu_affinity else self.cpusPerTask,  # cpu_limit
                        self.memoryPerTask,      # memory_limit
                        True                     # monitor_resources
                    )
                    for idx in range(i, batch_end)
                ]
                
                batch_duration = time.time() - batch_start_time
                log_batch_progress(batch_num, i, batch_end, batch_duration)
                
                batch_results = pool.map(task_runner, batch_args)
                
                for task_id, res, time_task, path in batch_results:
                    solved = (res == "sat" or res == "unsat")
                    results[task_id] = (solved, time_task, res)
        
        finally:
            pool.close()
            pool.join()
            self.resource_monitor.stop_monitoring()
        
        # Verify all results
        assert all(r is not None for r in results), "Missing results detected"
        return results
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SMT benchmarks')
    
    # Solver configuration
    parser.add_argument('--solver', type=str, default='z3', help='Solver executable name or path')
    parser.add_argument('--solver-args', nargs='*', default=[], help='Additional solver arguments (e.g., --solver-args --option1 value1 --option2)')
    parser.add_argument('--strategy-path', type=str, default=None, help='Path to Z3 solving strategy file (will be added as --strategy <content>)')

    # Benchmark configuration
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for each benchmark')
    parser.add_argument('--benchmark-dir', type=str, required=True, help='Directory containing SMT2 benchmark files')
    
    # Resource configuration
    parser.add_argument('--cpus-per-task', type=int, required=True, help='Number of CPUs to use per process')
    parser.add_argument('--memory-per-task', type=int, default=None, help='Memory limit per task in MB (auto-calculated if not specified)')
    parser.add_argument('--disable-cpu-affinity', action='store_true', help='Disable CPU affinity setting')

    # Output configuration
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file for results')
    parser.add_argument('--monitor-output', type=str, default=None, help='Directory for monitoring output files')
    parser.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warning", "error"], help="Logging level")
    
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Build solver command
    solver_cmd = [args.solver] + args.solver_args
    
    # Add strategy if provided
    if args.strategy_path:
        strategy = parse_strategy_file(args.strategy_path)
        solver_cmd.extend(["--strategy", strategy])

    print(f"Solver command: {' '.join(solver_cmd)}")
    
    # Find benchmarks
    benchmark_dir = Path(args.benchmark_dir)
    benchmarks = list(benchmark_dir.rglob("*.smt2"))
    
    if not benchmarks:
        print(f"Error: No .smt2 files found in {args.benchmark_dir}")
        exit(1)
    
    print(f"Found {len(benchmarks)} benchmarks")
    
    # Run evaluation
    evaluator = SolverEvaluator(
        solver_cmd=solver_cmd,
        benchmark_lst=benchmarks,
        timeout=args.timeout,
        cpus_per_task=args.cpus_per_task,
        memory_per_task=args.memory_per_task,
        disable_cpu_affinity=args.disable_cpu_affinity,
        monitor_output_dir=args.monitor_output
    )
    
    results = evaluator.getResLst()
    print(f"Completed: {sum(1 for r in results if r[0])} solved")