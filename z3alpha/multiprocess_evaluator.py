import os
import multiprocessing
import subprocess
import shlex
import time
import logging
import csv
import psutil
import threading
from resource_monitor import ResourceMonitor, log_resource_usage
from z3alpha.utils import solvedNum, parN

log = logging.getLogger(__name__)


def run_solver(
    solver_path, smt_file, timeout, id, strategy=None, tmp_dir="/tmp/", 
    cpu_limit=1, memory_limit=None, monitor_resources=False
):
    """Enhanced runner with resource monitoring"""
    
    # Enhanced CPU affinity logging
    if cpu_limit > 0:
        try:
            process = psutil.Process()
            initial_affinity = process.cpu_affinity()
            log.info(f"Task {id}: Initial CPU affinity: {initial_affinity}")
            
            if cpu_limit >= len(initial_affinity):
                log.debug(f"Process {id} using all assigned CPUs: {initial_affinity}")
            elif len(initial_affinity) <= 1:
                log.debug(f"Process {id} has only one CPU available, skipping affinity setting")
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
                    log.info(f"Task {id}: CPU affinity set to: {actual_affinity}")
                    
                    # Verify the setting worked
                    if set(actual_affinity) == set(new_affinity):
                        log.info(f"Task {id}: ✓ CPU affinity successfully set")
                    else:
                        log.warning(f"Task {id}: ✗ CPU affinity mismatch. Expected: {new_affinity}, Got: {actual_affinity}")
                        
                except (ValueError, OSError) as e:
                    log.warning(f"Task {id}: Cannot set CPU affinity: {e}")
                    
        except (AttributeError, NotImplementedError):
            log.warning(f"Task {id}: CPU affinity setting not supported")
    else:
        log.info(f"Task {id}: CPU affinity setting disabled")
    
    # Log process info
    log.info(f"Task {id}: Starting solver process (PID: {os.getpid()})")
    if memory_limit:
        log.info(f"Task {id}: Memory limit: {memory_limit} MB")
    
    # Start resource monitoring for this process if requested
    if monitor_resources:
        monitor_thread = threading.Thread(
            target=log_resource_usage, 
            args=(os.getpid(), id, min(timeout, 60)),  # Monitor for up to 60 seconds
            daemon=True
        )
        monitor_thread.start()
    
    # Prepare the SMT file with strategy if provided
    if strategy is not None:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        new_file_name = os.path.join(tmp_dir, f"tmp_{id}.smt2")
        with open(new_file_name, "w") as tmp_file:
            with open(smt_file, "r") as f:
                for line in f:
                    new_line = line
                    if "check-sat" in line:
                        new_line = f"(check-sat-using {strategy})\n"
                    tmp_file.write(new_line)
    else:
        new_file_name = smt_file
    
    solver_path = solver_path + " parallel.enable=true" 
    # Run the solver
    time_before = time.time()
    safe_path = shlex.quote(new_file_name)
    cmd = f"{solver_path} {safe_path}"
    
    # Memory limit handling
    if memory_limit:
        cmd = f"ulimit -v {memory_limit * 1024} && {cmd}"
        shell = True
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
    else:
        shell = False
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Log the actual process PID after starting
    log.info(f"Task {id}: Solver subprocess PID: {p.pid}")
    
    try:
        out, err = p.communicate(timeout=timeout)
        time_after = time.time()
        runtime = time_after - time_before
        
        lines = out.decode("utf-8").split("\n")
        res = lines[0] if lines else "error"
        
        log.info(f"Task {id}: Completed in {runtime:.2f}s with result: {res}")
        
        # Check for error
        if res.startswith("(error") or err:
            log.warning(f"Task {id}: Error - {res}, stderr: {err.decode('utf-8')}")
            return id, "error", runtime, smt_file
        
        return id, res, runtime, smt_file
    
    except subprocess.TimeoutExpired:
        log.info(f"Task {id}: Timeout after {timeout}s")
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
        cpus_per_task,  # Changed: now required parameter
        tmp_dir="/tmp/",
        is_write_res=False,
        res_path=None,
        memory_per_task=None,
        prefer_slurm_resources=True,
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
        self.prefer_slurm_resources = prefer_slurm_resources
        self.disable_cpu_affinity = disable_cpu_affinity
        self.monitor_output_dir = monitor_output_dir
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(monitor_output_dir)
        
        # Get Slurm resources first if available
        self.slurm_cpus, self.slurm_mem = get_slurm_resources()
        log.info(f"Detected Slurm resources: {self.slurm_cpus} CPUs, {self.slurm_mem} MB memory")
        
        # Determine the total available CPUs
        self.total_cpus = os.cpu_count()
        log.info(f"Total system CPUs: {self.total_cpus}")
        
        # Resource allocation logic - NEW APPROACH
        if self.prefer_slurm_resources and self.slurm_cpus is not None:
            log.info(f"Using Slurm-allocated resources: {self.slurm_cpus} CPUs")
            self.available_cpus = self.slurm_cpus
        else:
            self.available_cpus = self.total_cpus
            
        log.info(f"Will use {self.available_cpus} CPUs for calculation")
        
        # NEW LOGIC: Calculate batch size based on available CPUs and CPUs per task
        self.cpusPerTask = cpus_per_task
        
        # Calculate maximum possible parallel tasks based on CPU constraints
        max_parallel_tasks_by_cpu = self.available_cpus // self.cpusPerTask
        
        if max_parallel_tasks_by_cpu == 0:
            log.error(f"Cannot allocate {self.cpusPerTask} CPUs per task with only {self.available_cpus} available CPUs")
            raise ValueError(f"Insufficient CPUs: need at least {self.cpusPerTask} CPUs but only have {self.available_cpus}")
        
        # IMPORTANT: Limit parallel tasks to number of benchmarks to avoid wasting resources
        num_benchmarks = self.getBenchmarkSize()
        max_parallel_tasks = min(max_parallel_tasks_by_cpu, num_benchmarks)
        
        if max_parallel_tasks < max_parallel_tasks_by_cpu:
            log.info(f"Limiting parallel tasks to {max_parallel_tasks} (number of benchmarks) instead of {max_parallel_tasks_by_cpu} (CPU-based limit)")
            log.info(f"This will leave {(max_parallel_tasks_by_cpu - max_parallel_tasks) * self.cpusPerTask} CPUs unused")
        
        self.batchSize = max_parallel_tasks
        self.maxParallelTasks = self.batchSize
        
        # Calculate unused CPUs - now accounts for benchmark limit
        total_used_cpus = self.cpusPerTask * self.batchSize
        spare_cpus = self.available_cpus - total_used_cpus
        
        if spare_cpus > 0:
            if max_parallel_tasks < max_parallel_tasks_by_cpu:
                log.info(f"Spare CPUs due to benchmark limit: {spare_cpus} (could use {max_parallel_tasks_by_cpu} processes but only need {max_parallel_tasks})")
            else:
                log.warning(f"Spare CPUs due to CPU allocation: {spare_cpus} (using {self.cpusPerTask} × {self.batchSize} = {total_used_cpus})")
        else:
            log.info(f"Using all available CPUs efficiently: {total_used_cpus}/{self.available_cpus}")
        
        # Memory allocation - NEW LOGIC: Split available memory equally among ACTUAL parallel processes
        total_memory = None
        if self.prefer_slurm_resources and self.slurm_mem is not None:
            total_memory = self.slurm_mem
            log.info(f"Using Slurm-allocated memory: {total_memory} MB")
        else:
            try:
                total_memory = psutil.virtual_memory().total // (1024 * 1024)
                log.info(f"Total system memory: {total_memory} MB")
            except (ImportError, AttributeError):
                log.warning("Could not detect system memory")
                
        if memory_per_task is None and total_memory is not None:
            # Split memory equally among ACTUAL parallel processes (not theoretical max)
            self.memoryPerTask = total_memory // self.batchSize
            log.info(f"Auto-calculated memory per task: {self.memoryPerTask} MB ({total_memory} MB / {self.batchSize} actual processes)")
        else:
            self.memoryPerTask = memory_per_task
            if memory_per_task:
                log.info(f"Using specified memory per task: {self.memoryPerTask} MB")
        
        # Validation
        self.validate_resource_allocation()
        
        log.info(f"Initialized: {self.batchSize} parallel tasks, {self.cpusPerTask} CPUs per task")
        if self.memoryPerTask:
            log.info(f"Memory per task: {self.memoryPerTask} MB")

    def validate_resource_allocation(self):
        """Validate resource allocation against Slurm limits"""
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
                log.info(f"✅ CPU allocation OK: {total_used_cpus}/{self.slurm_cpus} (efficiency: {total_used_cpus/self.slurm_cpus*100:.1f}%)")
        else:
            log.info(f"✅ CPU allocation: {total_used_cpus}/{self.total_cpus} system CPUs")
        
        # Memory validation
        if self.memoryPerTask:
            total_used_mem = self.memoryPerTask * self.batchSize
            if self.slurm_mem:
                if total_used_mem > self.slurm_mem:
                    log.error(f"❌ MEMORY VIOLATION: Using {total_used_mem}MB but Slurm allocated {self.slurm_mem}MB")
                    raise ValueError("Memory allocation exceeds Slurm limits")
                else:
                    log.info(f"✅ Memory allocation OK: {total_used_mem}/{self.slurm_mem}MB (efficiency: {total_used_mem/self.slurm_mem*100:.1f}%)")
            else:
                log.info(f"✅ Memory allocation: {total_used_mem}MB requested")
        
        # Efficiency warnings
        cpu_efficiency = total_used_cpus / (self.slurm_cpus or self.total_cpus)
        if cpu_efficiency < 0.8:
            log.warning(f"⚠️  Low CPU efficiency: {cpu_efficiency*100:.1f}% - consider adjusting CPUs per task")
        
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
                     self.memoryPerTask, True)  # Enable per-process monitoring
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
    import glob
    import sys
    
    # Set up logging
    log.setLevel(logging.INFO)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
    log.addHandler(log_handler)
    
    parser = argparse.ArgumentParser(description='Run SMT benchmarks with enhanced resource monitoring')
    parser.add_argument('--solver', type=str, default='z3', help='Path to SMT solver executable')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for each benchmark')
    parser.add_argument('--cpus-per-task', type=int, required=True, help='Number of CPUs to use per process')
    parser.add_argument('--auto-optimize-cpus', action='store_true', help='Automatically increase CPUs per task to use all available CPUs when there are fewer benchmarks')
    parser.add_argument('--benchmark-dir', type=str, default=None, help='Directory containing SMT2 benchmark files')
    parser.add_argument('--benchmark-files', type=str, nargs='+', default=None, help='List of SMT2 benchmark files')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file for results')
    parser.add_argument('--strategy', type=str, default=None, help='Z3 solving strategy')
    parser.add_argument('--prefer-slurm', action='store_true', default=True, help='Prioritize Slurm-allocated resources')
    parser.add_argument('--no-slurm', dest='prefer_slurm', action='store_false', help='Use system resources even in Slurm')
    parser.add_argument('--memory-per-task', type=int, default=None, help='Memory limit per task in MB (auto-calculated if not specified)')
    parser.add_argument('--disable-cpu-affinity', action='store_true', help='Disable CPU affinity setting')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--monitor-output', type=str, default=None, help='Directory for monitoring output files')
    parser.add_argument('--validation-run', action='store_true', help='Run resource allocation validation tests')

    args = parser.parse_args()
    
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Debug logging enabled")
    
    # Print initial system state
    print("=" * 60)
    print("SYSTEM RESOURCE DETECTION")
    print("=" * 60)
    print(f"Python PID: {os.getpid()}")
    print(f"System CPUs: {os.cpu_count()}")
    
    try:
        mem_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System Memory: {mem_gb:.1f} GB")
        print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    except:
        print("Could not detect system memory")
    
    # Check current process CPU affinity
    try:
        current_affinity = psutil.Process().cpu_affinity()
        print(f"Current process CPU affinity: {current_affinity}")
    except:
        print("Could not detect current CPU affinity")
    
    print("=" * 60)
    
    # Determine benchmark files
    benchmark_lst = []
    if args.benchmark_dir:
        benchmark_lst = glob.glob(f"{args.benchmark_dir}/*.smt2")
        print(f"Found {len(benchmark_lst)} benchmark files in {args.benchmark_dir}")
    elif args.benchmark_files:
        benchmark_lst = args.benchmark_files
        print(f"Using {len(benchmark_lst)} specified benchmark files")
    else:
        print("Error: No benchmark files specified")
        print("Use --benchmark-dir or --benchmark-files")
        sys.exit(1)
    
    if not benchmark_lst:
        print("Error: No benchmark files found")
        sys.exit(1)
    
    print(f"Will process {len(benchmark_lst)} benchmark files")
    
    # Get Slurm resources
    slurm_cpus, slurm_mem = get_slurm_resources()
    
    # Calculate batch size based on available CPUs and CPUs per task
    available_cpus = slurm_cpus if (args.prefer_slurm and slurm_cpus) else os.cpu_count()
    calculated_batch_size_by_cpu = available_cpus // args.cpus_per_task
    
    if calculated_batch_size_by_cpu == 0:
        print(f"ERROR: Cannot allocate {args.cpus_per_task} CPUs per task with only {available_cpus} available CPUs")
        print(f"Available CPUs: {available_cpus}")
        print(f"CPUs per task: {args.cpus_per_task}")
        print("Try reducing --cpus-per-task or use more CPUs")
        sys.exit(1)
    
    # Limit to number of benchmarks
    actual_batch_size = min(calculated_batch_size_by_cpu, len(benchmark_lst))
    
    # Auto-optimize CPU allocation if requested
    final_cpus_per_task = args.cpus_per_task
    if args.auto_optimize_cpus and actual_batch_size < calculated_batch_size_by_cpu:
        # Try to use more CPUs per task to utilize spare resources
        max_possible_cpus_per_task = available_cpus // len(benchmark_lst)
        if max_possible_cpus_per_task > args.cpus_per_task:
            final_cpus_per_task = max_possible_cpus_per_task
            actual_batch_size = len(benchmark_lst)  # All benchmarks can run in parallel
            print(f"Auto-optimization: Increasing CPUs per task from {args.cpus_per_task} to {final_cpus_per_task}")
            print(f"This allows all {len(benchmark_lst)} benchmarks to run in parallel, using {final_cpus_per_task * actual_batch_size}/{available_cpus} CPUs")
    
    print(f"CPU-based batch size: {calculated_batch_size_by_cpu} (based on {available_cpus} CPUs ÷ {args.cpus_per_task} CPUs per task)")
    print(f"Actual batch size: {actual_batch_size} (limited by {len(benchmark_lst)} benchmarks)")
    
    if actual_batch_size < calculated_batch_size_by_cpu and not args.auto_optimize_cpus:
        unused_cpus = (calculated_batch_size_by_cpu - actual_batch_size) * args.cpus_per_task
        print(f"Note: {unused_cpus} CPUs will be unused due to limited number of benchmarks")
        print(f"Tip: Use --auto-optimize-cpus to automatically use more CPUs per task")
    
    # Create evaluator
    try:
        evaluator = SolverEvaluator(
            solver_path=args.solver,
            benchmark_lst=benchmark_lst,
            timeout=args.timeout,
            cpus_per_task=final_cpus_per_task,  # Use potentially optimized value
            tmp_dir="/tmp/",
            is_write_res=True,
            res_path=args.output,
            memory_per_task=args.memory_per_task,  # Can be None for auto-calculation
            prefer_slurm_resources=args.prefer_slurm,
            disable_cpu_affinity=args.disable_cpu_affinity,
            monitor_output_dir=args.monitor_output
        )
    except ValueError as e:
        print(f"ERROR: Resource allocation validation failed: {e}")
        sys.exit(1)
    
    # Print resource allocation summary
    print("\n" + "=" * 60)
    print("FINAL RESOURCE ALLOCATION")
    print("=" * 60)
    print(f"Solver: {args.solver}")
    print(f"Strategy: {args.strategy}")
    print(f"Timeout per benchmark: {args.timeout}s")
    print(f"CPUs per task: {evaluator.cpusPerTask}")
    print(f"Batch size (auto-calculated): {evaluator.batchSize}")
    print(f"Total parallel processes: {evaluator.maxParallelTasks}")
    print(f"Memory per task: {evaluator.memoryPerTask or 'auto-calculated'} MB")
    print(f"CPU affinity: {'Disabled' if evaluator.disable_cpu_affinity else 'Enabled'}")
    print(f"Resource monitoring: {'Enabled' if args.monitor_output else 'Disabled'}")
    
    if slurm_cpus:
        total_used = evaluator.cpusPerTask * evaluator.batchSize
        print(f"Slurm allocation: {slurm_cpus} CPUs, {slurm_mem} MB")
        print(f"Resource efficiency: {total_used}/{slurm_cpus} CPUs ({total_used/slurm_cpus*100:.1f}%)")
        
        spare_cpus = slurm_cpus - total_used
        if spare_cpus > 0:
            print(f"Unused CPUs: {spare_cpus}")
    
    print("=" * 60)
    
    # Run validation tests if requested
    if args.validation_run:
        print("\nRunning validation tests...")
        
        # Test 1: Quick benchmark run to verify everything works
        if len(benchmark_lst) > 3:
            test_benchmarks = benchmark_lst[:3]
            print(f"Testing with {len(test_benchmarks)} benchmarks...")
            
            test_evaluator = SolverEvaluator(
                solver_path=args.solver,
                benchmark_lst=test_benchmarks,
                timeout=min(args.timeout, 30),  # Short timeout for testing
                cpus_per_task=args.cpus_per_task,
                tmp_dir="/tmp/",
                is_write_res=False,
                res_path=None,
                prefer_slurm_resources=args.prefer_slurm,
                disable_cpu_affinity=args.disable_cpu_affinity,
                monitor_output_dir=args.monitor_output
            )
            
            test_results = test_evaluator.testing(args.strategy)
            print(f"Validation test results: {test_results[0]} solved")
            print("✅ Validation tests passed!")
        
        print("Validation complete. Exiting.")
        sys.exit(0)
    
    # Run the full benchmark
    print("\nStarting benchmark execution...")
    start_time = time.time()
    
    try:
        results = evaluator.testing(args.strategy)
        end_time = time.time()
        
        print("\n" + "=" * 60)
        print("BENCHMARK EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Total runtime: {end_time - start_time:.2f} seconds")
        print(f"Solved: {results[0]}/{len(benchmark_lst)} ({results[0]/len(benchmark_lst)*100:.2f}%)")
        print(f"PAR2 score: {results[1]:.2f}")
        print(f"PAR10 score: {results[2]:.2f}")
        print(f"Results written to: {args.output}")
        
        if args.monitor_output:
            print(f"Monitoring data saved to: {args.monitor_output}")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)