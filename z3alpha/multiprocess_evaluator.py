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
from z3alpha.utils import solvedNum, parN

log = logging.getLogger(__name__)


def run_solver(
    solver_path, smt_file, timeout, id, strategy=None, tmp_dir="/tmp/", 
    cpu_limit=1, memory_limit=None, monitor_resources=False, quiet=False
):
    """Enhanced runner with resource monitoring - now using CLI entry point"""
    
    # Enhanced CPU affinity logging
    if cpu_limit > 0:
        try:
            process = psutil.Process()
            initial_affinity = process.cpu_affinity()
            if not quiet: log.info(f"Task {id}: Initial CPU affinity: {initial_affinity}")
            
            if cpu_limit >= len(initial_affinity):
                if not quiet: log.debug(f"Process {id} using all assigned CPUs: {initial_affinity}")
            elif len(initial_affinity) <= 1:
                if not quiet: log.debug(f"Process {id} has only one CPU available, skipping affinity setting")
            else:
                # Distribute CPUs evenly across processes
                start_idx = (id * cpu_limit) % len(initial_affinity)
                end_idx = start_idx + cpu_limit
                
                # Handle wraparound if needed
                if end_idx <= len(initial_affinity):
                    new_affinity = initial_affinity[start_idx:end_idx]
                else:
                    # Wraparound case
                    new_affinity = initial_affinity[start_idx:] + initial_affinity[:end_idx - len(initial_affinity)]
                
                try:
                    process.cpu_affinity(new_affinity)
                    actual_affinity = process.cpu_affinity()
                    if not quiet: log.info(f"Task {id}: CPU affinity set to: {actual_affinity}")
                    
                    # Verify the setting worked
                    if set(actual_affinity) == set(new_affinity):
                        if not quiet: log.info(f"Task {id}: ✓ CPU affinity successfully set")
                    else:
                        log.warning(f"Task {id}: ✗ CPU affinity mismatch. Expected: {new_affinity}, Got: {actual_affinity}")
                        
                except (ValueError, OSError) as e:
                    log.warning(f"Task {id}: Cannot set CPU affinity: {e}")
                    
        except (AttributeError, NotImplementedError):
            log.warning(f"Task {id}: CPU affinity setting not supported")
    else:
        if not quiet: log.info(f"Task {id}: CPU affinity setting disabled")
    
    # Log process info
    if not quiet: log.info(f"Task {id}: Starting solver process (PID: {os.getpid()})")
    if memory_limit:
        if not quiet: log.info(f"Task {id}: Memory limit: {memory_limit} MB")
    
    # Start resource monitoring for this process if requested
    if monitor_resources:
        monitor_thread = threading.Thread(
            target=log_resource_usage, 
            args=(os.getpid(), id, min(timeout, 60)),  # Monitor for up to 60 seconds
            daemon=True
        )
        monitor_thread.start()
    
    # Build command using the CLI entry point
    # Convert Path objects to strings
    cmd = ["z3alpha", str(smt_file), "--z3-path", str(solver_path), "--tmp-dir", str(tmp_dir)]
    
    # Add strategy if provided
    if strategy is not None:
        cmd.extend(["--strategy", str(strategy)])
    
    # Run the solver
    time_before = time.time()
    
    if not quiet: log.info(f"Task {id}: Running command: {' '.join(cmd)}")
    
    # Memory limit handling
    if memory_limit:
        # Use ulimit to set memory limit
        ulimit_cmd = f"ulimit -v {memory_limit * 1024} && {' '.join(shlex.quote(arg) for arg in cmd)}"
        p = subprocess.Popen(ulimit_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    else:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Log the actual process PID after starting
    if not quiet: log.info(f"Task {id}: Solver subprocess PID: {p.pid}")
    
    try:
        out, err = p.communicate(timeout=timeout)
        time_after = time.time()
        runtime = time_after - time_before
        
        lines = out.decode("utf-8").split("\n")
        res = lines[0] if lines and len(lines[0]) > 0 else "error"
        
        if not quiet: log.info(f"Task {id}: Completed in {runtime:.2f}s with result: {res}")
        
        # Check for error
        if res.startswith("(error") or err:
            log.warning(f"Task {id}: Error - {res}, stderr: {err.decode('utf-8')}")
            return id, "error", runtime, smt_file
        
        return id, res, runtime, smt_file
    
    except subprocess.TimeoutExpired:
        if not quiet: log.info(f"Task {id}: Timeout after {timeout}s")
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        
        return id, "timeout", timeout, smt_file
        
def task_runner(args):
    """Enhanced wrapper function with resource monitoring"""
    smt_file, id, solver_path, timeout, strategy, tmp_dir, cpu_limit, memory_limit, monitor_resources = args
    return run_solver(
        solver_path=solver_path,
        smt_file=smt_file,
        timeout=timeout,
        id=id,
        strategy=strategy,
        tmp_dir=tmp_dir,
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
        solver_path,
        benchmark_lst,
        timeout,
        cpus_per_task,
        tmp_dir="/tmp/",
        is_write_res=False,
        res_path=None,
        memory_per_task=None,
        disable_cpu_affinity=False,
        monitor_output_dir=None
    ):
        self.solverPath = solver_path
        self.benchmarkLst = benchmark_lst
        assert self.getBenchmarkSize() > 0
        self.timeout = timeout
        assert self.timeout > 0
        self.tmpDir = tmp_dir
        self.isWriteRes = is_write_res
        self.resPath = res_path
        self.disable_cpu_affinity = disable_cpu_affinity
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(monitor_output_dir)
        
        # Get Slurm resources first if available
        self.slurm_cpus, self.slurm_mem = get_slurm_resources()
        log.info(f"Detected Slurm resources: {self.slurm_cpus} CPUs, {self.slurm_mem} MB memory")
        
        # Determine the total available CPUs
        self.total_cpus = os.cpu_count()
        log.info(f"Total system CPUs: {self.total_cpus}")
        
        # Use Slurm resources if available, otherwise use system resources
        if self.slurm_cpus is not None:
            log.info(f"Using Slurm-allocated resources: {self.slurm_cpus} CPUs")
            self.available_cpus = self.slurm_cpus
        else:
            self.available_cpus = self.total_cpus
            
        log.info(f"Will use {self.available_cpus} CPUs for calculation")
        
        # Calculate batch size based on available CPUs and CPUs per task
        self.cpusPerTask = cpus_per_task
        
        # Calculate maximum possible parallel tasks based on CPU constraints
        max_parallel_tasks_by_cpu = self.available_cpus // self.cpusPerTask
        
        if max_parallel_tasks_by_cpu == 0:
            log.error(f"Cannot allocate {self.cpusPerTask} CPUs per task with only {self.available_cpus} available CPUs")
            raise ValueError(f"Insufficient CPUs: need at least {self.cpusPerTask} CPUs but only have {self.available_cpus}")
        
        # Get total available memory
        total_memory = None
        if self.slurm_mem is not None:
            total_memory = self.slurm_mem
            log.info(f"Using Slurm-allocated memory: {total_memory} MB")
        else:
            try:
                total_memory = psutil.virtual_memory().total // (1024 * 1024)
                log.info(f"Total system memory: {total_memory} MB")
            except (ImportError, AttributeError):
                log.warning("Could not detect system memory")
        
        # Calculate memory constraints
        max_parallel_tasks_by_memory = float('inf')  # No memory limit by default
        
        if memory_per_task is not None:
            # User specified memory per task - this becomes a constraint
            self.memoryPerTask = memory_per_task
            if total_memory is not None:
                max_parallel_tasks_by_memory = total_memory // self.memoryPerTask
                log.info(f"Memory constraint: {total_memory} MB ÷ {self.memoryPerTask} MB/task = {max_parallel_tasks_by_memory} max parallel tasks")
                
                if max_parallel_tasks_by_memory == 0:
                    log.error(f"Cannot allocate {self.memoryPerTask} MB per task with only {total_memory} MB available memory")
                    raise ValueError(f"Insufficient memory: need at least {self.memoryPerTask} MB per task but only have {total_memory} MB total")
            else:
                log.warning(f"Using specified memory per task ({self.memoryPerTask} MB) but cannot verify against total memory")
        else:
            # Auto-calculate memory per task after determining batch size
            self.memoryPerTask = None
        
        # Determine maximum parallel tasks considering BOTH CPU and memory constraints
        num_benchmarks = self.getBenchmarkSize()
        
        # The actual limit is the minimum of: CPU limit, memory limit, and number of benchmarks
        constraints = [max_parallel_tasks_by_cpu, num_benchmarks]
        constraint_names = ["CPU", "benchmark count"]
        
        if max_parallel_tasks_by_memory != float('inf'):
            constraints.append(max_parallel_tasks_by_memory)
            constraint_names.append("memory")
        
        max_parallel_tasks = min(constraints)
        limiting_factor_idx = constraints.index(max_parallel_tasks)
        limiting_factor = constraint_names[limiting_factor_idx]
        
        log.info(f"Resource constraints:")
        log.info(f"  - CPU constraint: {max_parallel_tasks_by_cpu} tasks ({self.available_cpus} CPUs ÷ {self.cpusPerTask} CPUs/task)")
        if max_parallel_tasks_by_memory != float('inf'):
            log.info(f"  - Memory constraint: {max_parallel_tasks_by_memory} tasks ({total_memory} MB ÷ {self.memoryPerTask} MB/task)")
        log.info(f"  - Benchmark constraint: {num_benchmarks} tasks")
        log.info(f"  - Limiting factor: {limiting_factor}")
        log.info(f"  - Final batch size: {max_parallel_tasks} parallel tasks")
        
        self.batchSize = max_parallel_tasks
        self.maxParallelTasks = self.batchSize
        
        # Auto-calculate memory per task if not specified
        if self.memoryPerTask is None and total_memory is not None:
            self.memoryPerTask = total_memory // self.batchSize
            log.info(f"Auto-calculated memory per task: {self.memoryPerTask} MB ({total_memory} MB ÷ {self.batchSize} tasks)")
        
        # Calculate resource utilization
        total_used_cpus = self.cpusPerTask * self.batchSize
        spare_cpus = self.available_cpus - total_used_cpus
        
        if spare_cpus > 0:
            reason = f"limited by {limiting_factor}"
            log.info(f"Spare CPUs: {spare_cpus} ({reason})")
        else:
            log.info(f"Using all available CPUs efficiently: {total_used_cpus}/{self.available_cpus}")
        
        if self.memoryPerTask and total_memory:
            total_used_memory = self.memoryPerTask * self.batchSize
            spare_memory = total_memory - total_used_memory
            if spare_memory > 0:
                log.info(f"Spare memory: {spare_memory} MB")
            else:
                log.info(f"Using all available memory efficiently: {total_used_memory}/{total_memory} MB")
        
        # Validation
        self.validate_resource_allocation()
        
        log.info(f"Initialized: {self.batchSize} parallel tasks, {self.cpusPerTask} CPUs per task")
        if self.memoryPerTask:
            log.info(f"Memory per task: {self.memoryPerTask} MB")

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

    def getResLst(self, strat_str):
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
                    (self.benchmarkLst[idx], idx, self.solverPath, self.timeout, strat_str, 
                     self.tmpDir, 0 if self.disable_cpu_affinity else self.cpusPerTask, 
                     self.memoryPerTask, True)
                    for idx in range(i, batch_end)
                ]
                
                batch_start_time = time.time()
                batch_results = pool.map(task_runner, batch_args)
                batch_duration = time.time() - batch_start_time
                
                log.info(f"Batch {i//self.batchSize + 1} completed in {batch_duration:.2f}s")
                
                for id, res, time_task, path in batch_results:
                    solved = True if (res == "sat" or res == "unsat") else False
                    results[id] = (solved, time_task, res)
                    
                # Brief pause between batches to reduce system load
                if batch_end < size:
                    time.sleep(1)
        
        finally:
            pool.close()
            pool.join()
            self.resource_monitor.stop_monitoring()
        
        # Verify all results
        for i in range(size):
            assert results[i] is not None, f"Missing result for task {i}"
            
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
    import sys
    
    # Set up logging
    log.setLevel(logging.INFO)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
    log.addHandler(log_handler)
    
    parser = argparse.ArgumentParser(description='Run SMT benchmarks')
    parser.add_argument('--solver', type=str, default='z3', help='Path to SMT solver executable')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for each benchmark')
    parser.add_argument('--cpus-per-task', type=int, required=True, help='Number of CPUs to use per process')
    parser.add_argument('--benchmark-dir', type=str, required=True, help='Directory containing SMT2 benchmark files')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file for results')
    parser.add_argument('--strategy-path', type=str, default=None, help='Path to Z3 solving strategy file')
    parser.add_argument('--memory-per-task', type=int, default=None, help='Memory limit per task in MB (auto-calculated if not specified)')
    parser.add_argument('--disable-cpu-affinity', action='store_true', help='Disable CPU affinity setting')
    parser.add_argument('--monitor-output', type=str, default=None, help='Directory for monitoring output files')

    args = parser.parse_args()

    # Determine strategy to use
    strategy = None

    try:
        with open(args.strategy_path, 'r') as f:
            strategy = f.read().strip()
        
        if not strategy:
            raise ValueError(f"Strategy file {args.strategy_path} is empty")
        
        log.info(f"Strategy loaded from {args.strategy_path}: {strategy[:100]}{'...' if len(strategy) > 100 else ''}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Strategy file not found: {args.strategy_path}")
    except Exception as e:
        raise Exception(f"Error reading strategy file {args.strategy_path}: {e}")

    # Find benchmark files recursively using Path
    benchmark_dir = Path(args.benchmark_dir)
    benchmark_lst = list(benchmark_dir.rglob("*.smt2"))
    
    if not benchmark_lst:
        print(f"Error: No .smt2 files found in {args.benchmark_dir} or its subdirectories")
        sys.exit(1)
    
    print(f"Found {len(benchmark_lst)} benchmark files in {args.benchmark_dir}")
    
    # Create evaluator and run
    try:
        evaluator = SolverEvaluator(
            solver_path=args.solver,
            benchmark_lst=benchmark_lst,
            timeout=args.timeout,
            cpus_per_task=args.cpus_per_task,
            tmp_dir="/tmp/",
            is_write_res=True,
            res_path=args.output,
            memory_per_task=args.memory_per_task,
            disable_cpu_affinity=args.disable_cpu_affinity,
            monitor_output_dir=args.monitor_output
        )
        
        print(f"Running benchmarks with {evaluator.batchSize} parallel processes...")
        results = evaluator.testing(strategy)
        
        print(f"\nResults:")
        print(f"Solved: {results[0]}/{len(benchmark_lst)} ({results[0]/len(benchmark_lst)*100:.2f}%)")
        print(f"PAR2 score: {results[1]:.2f}")
        print(f"PAR10 score: {results[2]:.2f}")
        print(f"Results written to: {args.output}")
        
        if args.monitor_output:
            print(f"Monitoring data saved to: {args.monitor_output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)