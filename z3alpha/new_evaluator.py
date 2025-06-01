import os
import multiprocessing
import subprocess
import shlex
import time
import logging
import csv
import psutil
import threading
import json
from datetime import datetime

from z3alpha.utils import solvedNum, parN

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
log.addHandler(log_handler)


class ResourceMonitor:
    """Real-time resource monitoring for solver processes"""
    
    def __init__(self, monitor_output_dir=None):
        self.monitor_output_dir = monitor_output_dir
        self.monitoring = False
        self.monitor_thread = None
        self.process_data = []
        self._process_objs = {}  # Persist Process objects per PID
        
        if self.monitor_output_dir:
            os.makedirs(self.monitor_output_dir, exist_ok=True)
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        log.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and save data"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        if self.monitor_output_dir and self.process_data:
            self._save_monitoring_data()
        
        log.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Monitor all running solver processes
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
                    try:
                        if proc.info['name'] in ['z3', 'z3-4.8.8'] or any('z3' in str(arg) for arg in proc.info['cmdline'] or []):
                            pid = proc.info['pid']
                            # Reuse Process object if possible
                            if pid not in self._process_objs:
                                self._process_objs[pid] = psutil.Process(pid)
                            process = self._process_objs[pid]
                            cpu_percent = process.cpu_percent(interval=None)  # Use None for non-blocking
                            memory_mb = process.memory_info().rss / (1024 * 1024)
                            
                            try:
                                cpu_affinity = process.cpu_affinity()
                            except (psutil.AccessDenied, AttributeError):
                                cpu_affinity = []
                            
                            process_info = {
                                'timestamp': current_time,
                                'pid': pid,
                                'cpu_percent': cpu_percent,
                                'memory_mb': round(memory_mb, 2),
                                'cpu_affinity': cpu_affinity,
                                'num_threads': process.num_threads(),
                                'status': process.status()
                            }
                            
                            self.process_data.append(process_info)
                            
                            # Log high resource usage
                            if cpu_percent > 90:
                                log.warning(f"High CPU usage: PID {pid} using {cpu_percent:.1f}% CPU")
                            if memory_mb > 500:  # > 500MB
                                log.warning(f"High memory usage: PID {pid} using {memory_mb:.1f} MB")
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
            
            time.sleep(3)  # Monitor every 3 seconds
    
    def _save_monitoring_data(self):
        """Save collected monitoring data"""
        if not self.process_data:
            return
        
        # Save as CSV
        csv_file = os.path.join(self.monitor_output_dir, "detailed_process_monitor.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'pid', 'cpu_percent', 'memory_mb', 
                                                  'cpu_affinity', 'num_threads', 'status'])
            writer.writeheader()
            writer.writerows(self.process_data)
        
        # Save as JSON for detailed analysis
        json_file = os.path.join(self.monitor_output_dir, "process_monitor.json")
        with open(json_file, 'w') as f:
            json.dump(self.process_data, f, indent=2, default=str)
        
        # Generate summary statistics
        self._generate_summary()
        
        log.info(f"Monitoring data saved to {self.monitor_output_dir}")
    
    def _generate_summary(self):
        """Generate monitoring summary statistics"""
        if not self.process_data:
            return
        
        summary_file = os.path.join(self.monitor_output_dir, "monitoring_summary.txt")
        
        # Calculate statistics
        cpu_values = [d['cpu_percent'] for d in self.process_data if d['cpu_percent'] > 0]
        memory_values = [d['memory_mb'] for d in self.process_data]
        unique_pids = set(d['pid'] for d in self.process_data)
        
        with open(summary_file, 'w') as f:
            f.write("RESOURCE MONITORING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Monitoring duration: {len(self.process_data)} data points\n")
            f.write(f"Unique processes monitored: {len(unique_pids)}\n")
            f.write(f"PIDs: {sorted(unique_pids)}\n\n")
            
            if cpu_values:
                f.write(f"CPU Usage Statistics:\n")
                f.write(f"  Average: {sum(cpu_values)/len(cpu_values):.2f}%\n")
                f.write(f"  Peak: {max(cpu_values):.2f}%\n")
                f.write(f"  Min: {min(cpu_values):.2f}%\n\n")
            
            if memory_values:
                f.write(f"Memory Usage Statistics:\n")
                f.write(f"  Average: {sum(memory_values)/len(memory_values):.2f} MB\n")
                f.write(f"  Peak: {max(memory_values):.2f} MB\n")
                f.write(f"  Min: {min(memory_values):.2f} MB\n\n")
            
            # CPU affinity analysis
            affinity_data = [d['cpu_affinity'] for d in self.process_data if d['cpu_affinity']]
            if affinity_data:
                f.write(f"CPU Affinity Analysis:\n")
                unique_affinities = set(tuple(sorted(aff)) for aff in affinity_data)
                for aff in unique_affinities:
                    count = sum(1 for a in affinity_data if tuple(sorted(a)) == aff)
                    f.write(f"  CPUs {aff}: {count} observations\n")

def log_resource_usage(pid, task_id, duration):
    """Log resource usage for a specific process"""
    try:
        process = psutil.Process(pid)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                cpu_percent = process.cpu_percent(interval=1)
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_affinity = process.cpu_affinity()
                
                log.debug(f"Task {task_id} (PID {pid}): CPU={cpu_percent:.1f}%, "
                         f"Memory={memory_mb:.1f}MB, Affinity={cpu_affinity}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            time.sleep(5)  # Log every 5 seconds
            
    except Exception as e:
        log.debug(f"Resource logging failed for task {task_id}: {e}")


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
            
            if cpu_limit >= len(current_affinity):
                log.debug(f"Process {id} using all assigned CPUs: {current_affinity}")
            elif len(current_affinity) <= 1:
                log.debug(f"Process {id} has only one CPU available, skipping affinity setting")
            else:
                # Distribute CPUs evenly across processes
                start_idx = (id * cpu_limit) % len(current_affinity)
                end_idx = start_idx + cpu_limit
                
                # Handle wraparound if needed
                if end_idx <= len(current_affinity):
                    new_affinity = current_affinity[start_idx:end_idx]
                else:
                    # Wraparound case
                    new_affinity = current_affinity[start_idx:] + current_affinity[:end_idx - len(current_affinity)]
                
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


class SolverEvaluator:
    def __init__(
        self,
        solver_path,
        benchmark_lst,
        timeout,
        batch_size,
        tmp_dir="/tmp/",
        is_write_res=False,
        res_path=None,
        cpus_per_task=None,
        memory_per_task=None,
        max_parallel_tasks=None,
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
        
        # Resource allocation logic (keeping your existing logic)
        if self.prefer_slurm_resources and self.slurm_cpus is not None:
            log.info(f"Using Slurm-allocated resources: {self.slurm_cpus} CPUs")
            self.available_cpus = self.slurm_cpus
        else:
            self.available_cpus = self.total_cpus
            
        log.info(f"Will use {self.available_cpus} CPUs for calculation")
        
        self.batchSize = max_parallel_tasks if max_parallel_tasks else batch_size
        
        if self.batchSize > self.available_cpus:
            log.warning(f"Batch size ({self.batchSize}) > available CPUs ({self.available_cpus})")
            log.warning(f"Reducing batch size to {self.available_cpus}")
            self.batchSize = self.available_cpus
        
        if cpus_per_task is None:
            self.cpusPerTask = max(1, self.available_cpus // self.batchSize)
        else:
            self.cpusPerTask = cpus_per_task
        
        spare_cpus = self.available_cpus - (self.cpusPerTask * self.batchSize)
        if spare_cpus > 0:
            log.warning(f"Spare CPUs: {spare_cpus} (using {self.cpusPerTask} × {self.batchSize} = {self.cpusPerTask * self.batchSize})")
        
        # Memory allocation
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
            self.memoryPerTask = total_memory // self.batchSize
            log.info(f"Calculated memory per task: {self.memoryPerTask} MB")
        else:
            self.memoryPerTask = memory_per_task
        
        self.maxParallelTasks = self.batchSize
        
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
            log.warning(f"⚠️  Low CPU efficiency: {cpu_efficiency*100:.1f}% - consider reducing batch size")
        
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


if __name__ == "__main__":
    import argparse
    import glob
    import sys
    
    parser = argparse.ArgumentParser(description='Run SMT benchmarks with enhanced resource monitoring')
    parser.add_argument('--solver', type=str, default='z3', help='Path to SMT solver executable')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for each benchmark')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of parallel tasks')
    parser.add_argument('--benchmark-dir', type=str, default=None, help='Directory containing SMT2 benchmark files')
    parser.add_argument('--benchmark-files', type=str, nargs='+', default=None, help='List of SMT2 benchmark files')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file for results')
    parser.add_argument('--strategy', type=str, default=None, help='Z3 solving strategy')
    parser.add_argument('--prefer-slurm', action='store_true', default=True, help='Prioritize Slurm-allocated resources')
    parser.add_argument('--no-slurm', dest='prefer_slurm', action='store_false', help='Use system resources even in Slurm')
    parser.add_argument('--max-cpus', type=int, default=None, help='Manually set maximum CPUs to use')
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
    
    """
    # Limit benchmarks for testing if there are too many
    if len(benchmark_lst) > 20 and args.timeout < 60:
        print(f"Limiting to first 20 benchmarks for quick testing (found {len(benchmark_lst)})")
        benchmark_lst = benchmark_lst[:20]
    """
    
    print(f"Will process {len(benchmark_lst)} benchmark files")
    
    # Get Slurm resources
    slurm_cpus, slurm_mem = get_slurm_resources()
    
    # Override CPU count if specified
    cpus_per_task = None
    if args.max_cpus:
        print(f"Manually limiting to {args.max_cpus} CPUs as specified by --max-cpus")
        adjusted_batch_size = min(args.batch_size, args.max_cpus)
        cpus_per_task = args.max_cpus // adjusted_batch_size
    
    # Create evaluator
    try:
        evaluator = SolverEvaluator(
            solver_path=args.solver,
            benchmark_lst=benchmark_lst,
            timeout=args.timeout,
            batch_size=args.batch_size,
            tmp_dir="/tmp/",
            is_write_res=True,
            res_path=args.output,
            cpus_per_task=cpus_per_task,
            memory_per_task=None,
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
    print(f"Batch size: {evaluator.batchSize}")
    print(f"CPUs per task: {evaluator.cpusPerTask}")
    print(f"Memory per task: {evaluator.memoryPerTask or 'unlimited'}")
    print(f"CPU affinity: {'Disabled' if evaluator.disable_cpu_affinity else 'Enabled'}")
    print(f"Resource monitoring: {'Enabled' if args.monitor_output else 'Disabled'}")
    
    if slurm_cpus:
        total_used = evaluator.cpusPerTask * evaluator.batchSize
        print(f"Slurm allocation: {slurm_cpus} CPUs, {slurm_mem} MB")
        print(f"Resource efficiency: {total_used}/{slurm_cpus} CPUs ({total_used/slurm_cpus*100:.1f}%)")
    
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
                batch_size=min(args.batch_size, 2),
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