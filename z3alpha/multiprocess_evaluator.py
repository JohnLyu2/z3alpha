import os
import multiprocessing
import subprocess
import shlex
import time
import logging
import csv
import psutil
import threading
from pathlib import Path
from z3alpha.resource_monitor import ResourceMonitor, log_resource_usage
from z3alpha.utils import solvedNum, parN, setup_logging
from z3alpha.resource_allocation import set_cpu_affinity, calculate_resource_allocation

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

def get_slurm_resources():
    """Get resources allocated by Slurm if running in a Slurm environment"""
    cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
    if cpus_per_task:
        cpus_per_task = int(cpus_per_task)
        log.info(f"Found SLURM_CPUS_PER_TASK: {cpus_per_task}")
    else:
        cpus_on_node = os.environ.get('SLURM_CPUS_ON_NODE')
        if cpus_on_node:
            cpus_per_task = int(cpus_on_node)
            log.info(f"Found SLURM_CPUS_ON_NODE: {cpus_per_task}")
        else:
            cpus_per_task = None
            log.info("No Slurm CPU allocation detected")
    
    # Get memory allocation
    total_mem_mb = None
    mem_per_node = os.environ.get('SLURM_MEM_PER_NODE')
    if mem_per_node:
        if mem_per_node.endswith('G'):
            total_mem_mb = int(float(mem_per_node[:-1]) * 1024)
        elif mem_per_node.endswith('M'):
            total_mem_mb = int(mem_per_node[:-1])
        elif mem_per_node.endswith('K'):
            total_mem_mb = int(float(mem_per_node[:-1]) / 1024)
        else:
            total_mem_mb = int(mem_per_node)
        log.info(f"Found SLURM_MEM_PER_NODE: {mem_per_node} -> {total_mem_mb} MB")
    
    if total_mem_mb is None:
        mem_per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
        if mem_per_cpu and cpus_per_task:
            if mem_per_cpu.endswith('G'):
                mem_per_cpu_mb = int(float(mem_per_cpu[:-1]) * 1024)
            elif mem_per_cpu.endswith('M'):
                mem_per_cpu_mb = int(mem_per_cpu[:-1])
            elif mem_per_cpu.endswith('K'):
                mem_per_cpu_mb = int(float(mem_per_cpu[:-1]) / 1024)
            else:
                mem_per_cpu_mb = int(mem_per_cpu)
            
            total_mem_mb = mem_per_cpu_mb * cpus_per_task
            log.info(f"Found SLURM_MEM_PER_CPU: {mem_per_cpu} -> {total_mem_mb} MB")
    
    is_slurm = False
    if 'SLURM_JOB_ID' in os.environ:
        is_slurm = True
        log.info(f"Running in Slurm job {os.environ.get('SLURM_JOB_ID')}")
        
        if cpus_per_task is None:
            cpus_per_task = os.cpu_count()
            log.warning(f"Running in Slurm but could not detect CPU allocation, using system CPU count: {cpus_per_task}")
    
    log.info(f"Final Slurm resources: {cpus_per_task} CPUs, {total_mem_mb} MB memory")
    
    return cpus_per_task, total_mem_mb


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
        allocation.log_summary()
        
        # Store allocation results
        self.batchSize = allocation.batch_size
        self.maxParallelTasks = allocation.batch_size
        self.cpusPerTask = allocation.cpus_per_task
        self.memoryPerTask = allocation.memory_per_task

    def validate_resource_allocation(self):
        """Validate resource allocation against system limits"""
        log.info("=" * 50)
        log.info("RESOURCE ALLOCATION VALIDATION")
        log.info("=" * 50)
        
        # CPU validation
        total_used_cpus = self.cpusPerTask * self.batchSize
        if self.slurm_cpus:
            if total_used_cpus > self.slurm_cpus:
                log.error(f"❌ CPU VIOLATION: Using {total_used_cpus} but Slurm allocated {self.slurm_cpus}")
                raise ValueError("CPU allocation exceeds Slurm limits")
            else:
                cpu_efficiency = total_used_cpus / self.slurm_cpus
                log.info(f"✅ CPU allocation OK: {total_used_cpus}/{self.slurm_cpus} (efficiency: {cpu_efficiency*100:.1f}%)")
        else:
            cpu_efficiency = total_used_cpus / self.total_cpus
            log.info(f"✅ CPU allocation: {total_used_cpus}/{self.total_cpus} system CPUs (efficiency: {cpu_efficiency*100:.1f}%)")
        
        # Memory validation
        if self.memoryPerTask:
            total_used_mem = self.memoryPerTask * self.batchSize
            total_available_mem = self.slurm_mem or (psutil.virtual_memory().total // (1024 * 1024) if hasattr(psutil, 'virtual_memory') else None)
            
            if total_available_mem:
                if total_used_mem > total_available_mem:
                    log.error(f"❌ MEMORY VIOLATION: Using {total_used_mem}MB but only {total_available_mem}MB available")
                    raise ValueError("Memory allocation exceeds available memory")
                else:
                    mem_efficiency = total_used_mem / total_available_mem
                    log.info(f"✅ Memory allocation OK: {total_used_mem}/{total_available_mem}MB (efficiency: {mem_efficiency*100:.1f}%)")
            else:
                log.info(f"✅ Memory allocation: {total_used_mem}MB requested (cannot verify total)")
        
        log.info("=" * 50)

    def getBenchmarkSize(self):
        return len(self.benchmarkLst)

    def getResLst(self):
        size = self.getBenchmarkSize()
        results = [None] * size
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        log.info(f"Starting parallel execution: {self.maxParallelTasks} processes")
        log.info(f"Resource allocation per process: {self.cpusPerTask} CPUs, {self.memoryPerTask or 'unlimited'} MB")
        
        pool = multiprocessing.Pool(processes=self.maxParallelTasks)
        
        try:
            for i in range(0, size, self.batchSize):
                batch_end = min(i + self.batchSize, size)
                
                log.info(f"Processing batch {i//self.batchSize + 1}: tasks {i} to {batch_end-1}")
                
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

    def testing(self, strat_str):
        log.info("=" * 60)
        log.info("STARTING SOLVER EVALUATION WITH RESOURCE MONITORING")
        log.info("=" * 60)
        
        start_time = time.time()
        results = self.getResLst(strat_str)
        total_time = time.time() - start_time
        
        log.info(f"Evaluation completed in {total_time:.2f} seconds")
        
        if self.isWriteRes:
            with open(self.resPath, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "path", "solved", "time", "result"])
                for i in range(len(self.benchmarkLst)):
                    writer.writerow([
                        i,
                        self.benchmarkLst[i],
                        results[i][0],
                        results[i][1],
                        results[i][2],
                    ])
            log.info(f"Results written to: {self.resPath}")
        
        solved = solvedNum(results)
        par2 = parN(results, 2, self.timeout)
        par10 = parN(results, 10, self.timeout)
        
        # Performance summary
        log.info("=" * 60)
        log.info("EVALUATION RESULTS")
        log.info("=" * 60)
        log.info(f"Total benchmarks: {len(self.benchmarkLst)}")
        log.info(f"Solved: {solved} ({solved/len(self.benchmarkLst)*100:.1f}%)")
        log.info(f"PAR2 score: {par2:.2f}")
        log.info(f"PAR10 score: {par10:.2f}")
        log.info(f"Total execution time: {total_time:.2f}s")
        log.info(f"Average time per benchmark: {total_time/len(self.benchmarkLst):.2f}s")
        log.info("=" * 60)
        
        return (solved, par2, par10)
    
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