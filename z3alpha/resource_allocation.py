import os
import logging
import psutil

log = logging.getLogger(__name__)


# ============================================================================
# CPU AFFINITY MANAGEMENT
# ============================================================================

def set_cpu_affinity(process, task_id, cpu_limit, initial_affinity):
    """
    Set CPU affinity for a process.
    
    Args:
        process: psutil.Process object
        task_id: Task identifier for logging
        cpu_limit: Number of CPUs to allocate
        initial_affinity: List of initially available CPU cores
    
    Returns:
        bool: True if affinity was set successfully, False otherwise
    """
    if cpu_limit <= 0:
        log.debug(f"Task {task_id}: CPU affinity setting disabled")
        return False
    
    try:
        log.debug(f"Task {task_id}: Initial CPU affinity: {initial_affinity}")
        
        # Check if we need to set affinity
        if cpu_limit >= len(initial_affinity):
            log.debug(f"Task {task_id}: Using all assigned CPUs: {initial_affinity}")
            return True
            
        if len(initial_affinity) <= 1:
            log.debug(f"Task {task_id}: Only one CPU available, skipping affinity setting")
            return True
        
        # Distribute CPUs evenly across processes
        start_idx = (task_id * cpu_limit) % len(initial_affinity)
        end_idx = start_idx + cpu_limit
        
        # Handle wraparound if needed
        if end_idx <= len(initial_affinity):
            new_affinity = initial_affinity[start_idx:end_idx]
        else:
            new_affinity = initial_affinity[start_idx:] + initial_affinity[:end_idx - len(initial_affinity)]
        
        # Set and verify affinity
        process.cpu_affinity(new_affinity)
        actual_affinity = process.cpu_affinity()
        
        if set(actual_affinity) == set(new_affinity):
            log.info(f"Task {task_id}: CPU affinity set to {actual_affinity}")
            return True
        else:
            log.warning(f"Task {task_id}: CPU affinity mismatch. Expected: {new_affinity}, Got: {actual_affinity}")
            return False
            
    except (ValueError, OSError) as e:
        log.warning(f"Task {task_id}: Cannot set CPU affinity: {e}")
        return False
    except (AttributeError, NotImplementedError):
        log.warning(f"Task {task_id}: CPU affinity not supported on this platform")
        return False


# ============================================================================
# SLURM RESOURCE DETECTION
# ============================================================================

def get_slurm_resources():
    """
    Detect resources allocated by Slurm.
    
    Returns:
        tuple: (cpus, memory_mb, is_slurm)
            - cpus: Number of CPUs allocated (or None)
            - memory_mb: Memory in MB allocated (or None)
            - is_slurm: Whether running in Slurm environment
    """
    is_slurm = 'SLURM_JOB_ID' in os.environ
    
    # Detect CPUs
    cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if cpus:
        cpus = int(cpus)
        log.debug(f"Found SLURM_CPUS_PER_TASK: {cpus}")
    else:
        cpus_on_node = os.environ.get('SLURM_CPUS_ON_NODE')
        if cpus_on_node:
            cpus = int(cpus_on_node)
            log.debug(f"Found SLURM_CPUS_ON_NODE: {cpus}")
    
    # Detect memory
    memory_mb = None
    mem_per_node = os.environ.get('SLURM_MEM_PER_NODE')
    
    if mem_per_node:
        memory_mb = _parse_memory_string(mem_per_node)
        log.debug(f"Found SLURM_MEM_PER_NODE: {mem_per_node} -> {memory_mb} MB")
    else:
        mem_per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
        if mem_per_cpu and cpus:
            mem_per_cpu_mb = _parse_memory_string(mem_per_cpu)
            memory_mb = mem_per_cpu_mb * cpus
            log.debug(f"Found SLURM_MEM_PER_CPU: {mem_per_cpu} -> {memory_mb} MB total")
    
    if is_slurm:
        log.info(f"Running in Slurm job {os.environ.get('SLURM_JOB_ID')}")
        if cpus is None:
            cpus = os.cpu_count()
            log.warning(f"Running in Slurm but could not detect CPU allocation, using system count: {cpus}")
    
    log.info(f"Detected Slurm resources: {cpus} CPUs, {memory_mb} MB")
    return cpus, memory_mb, is_slurm


def _parse_memory_string(mem_str):
    """Parse memory string like '16G', '1024M', '512K' to MB"""
    if mem_str.endswith('G'):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith('M'):
        return int(mem_str[:-1])
    elif mem_str.endswith('K'):
        return int(float(mem_str[:-1]) / 1024)
    else:
        return int(mem_str)


# ============================================================================
# RESOURCE ALLOCATION CALCULATION
# ============================================================================

class ResourceAllocation:
    """Container for resource allocation results"""
    def __init__(self, batch_size, cpus_per_task, memory_per_task, 
                 limiting_factor, total_cpus, total_memory):
        self.batch_size = batch_size
        self.cpus_per_task = cpus_per_task
        self.memory_per_task = memory_per_task
        self.limiting_factor = limiting_factor
        self.total_cpus = total_cpus
        self.total_memory = total_memory
    
    def validate(self):
        """Validate resource allocation doesn't exceed limits"""
        log.info("=" * 50)
        log.info("RESOURCE ALLOCATION VALIDATION")
        log.info("=" * 50)
        
        # CPU validation
        total_used_cpus = self.cpus_per_task * self.batch_size
        cpu_efficiency = total_used_cpus / self.total_cpus
        
        if total_used_cpus > self.total_cpus:
            log.error(f"❌ CPU VIOLATION: Using {total_used_cpus} but only {self.total_cpus} available")
            raise ValueError("CPU allocation exceeds available CPUs")
        
        log.info(f"✅ CPU allocation: {total_used_cpus}/{self.total_cpus} (efficiency: {cpu_efficiency*100:.1f}%)")
        
        # Memory validation
        if self.memory_per_task and self.total_memory:
            total_used_mem = self.memory_per_task * self.batch_size
            mem_efficiency = total_used_mem / self.total_memory
            
            if total_used_mem > self.total_memory:
                log.error(f"❌ MEMORY VIOLATION: Using {total_used_mem}MB but only {self.total_memory}MB available")
                raise ValueError("Memory allocation exceeds available memory")
            
            log.info(f"✅ Memory allocation: {total_used_mem}/{self.total_memory}MB (efficiency: {mem_efficiency*100:.1f}%)")
        elif self.memory_per_task:
            log.info(f"✅ Memory per task: {self.memory_per_task}MB (total unknown)")
        
        log.info("=" * 50)
    
    def log_summary(self):
        """Log resource allocation summary"""
        log.info("=" * 60)
        log.info("RESOURCE ALLOCATION SUMMARY")
        log.info("=" * 60)
        log.info(f"Parallel tasks: {self.batch_size}")
        log.info(f"CPUs per task: {self.cpus_per_task}")
        log.info(f"Total CPU usage: {self.cpus_per_task * self.batch_size}/{self.total_cpus}")
        
        if self.memory_per_task:
            log.info(f"Memory per task: {self.memory_per_task} MB")
            if self.total_memory:
                log.info(f"Total memory usage: {self.memory_per_task * self.batch_size}/{self.total_memory} MB")
        
        log.info(f"Limiting factor: {self.limiting_factor}")
        
        # Calculate spare resources
        spare_cpus = self.total_cpus - (self.cpus_per_task * self.batch_size)
        if spare_cpus > 0:
            log.info(f"Spare CPUs: {spare_cpus}")
        
        if self.memory_per_task and self.total_memory:
            spare_mem = self.total_memory - (self.memory_per_task * self.batch_size)
            if spare_mem > 0:
                log.info(f"Spare memory: {spare_mem} MB")
        
        log.info("=" * 60)


def calculate_resource_allocation(cpus_per_task, memory_per_task, num_benchmarks):
    """
    Calculate optimal resource allocation for parallel task execution.
    
    Args:
        cpus_per_task: Number of CPUs to allocate per task
        memory_per_task: Memory in MB per task (None for auto-calculate)
        num_benchmarks: Total number of benchmarks to run
    
    Returns:
        ResourceAllocation: Object containing allocation details
    """
    # Get Slurm and system resources
    slurm_cpus, slurm_mem, is_slurm = get_slurm_resources()
    total_cpus = os.cpu_count()
    
    # Determine available CPUs
    available_cpus = slurm_cpus if slurm_cpus else total_cpus
    log.info(f"Available CPUs: {available_cpus} (system: {total_cpus})")
    
    # Get total memory
    total_memory = None
    if slurm_mem:
        total_memory = slurm_mem
        log.info(f"Available memory: {total_memory} MB (Slurm)")
    else:
        try:
            total_memory = psutil.virtual_memory().total // (1024 * 1024)
            log.info(f"Available memory: {total_memory} MB (system)")
        except (ImportError, AttributeError):
            log.warning("Could not detect system memory")
    
    # Calculate constraints
    max_by_cpu = available_cpus // cpus_per_task
    max_by_benchmarks = num_benchmarks
    
    if max_by_cpu == 0:
        raise ValueError(f"Insufficient CPUs: need {cpus_per_task} per task but only have {available_cpus}")
    
    constraints = {
        'CPU': max_by_cpu,
        'benchmark count': max_by_benchmarks
    }
    
    # Add memory constraint if specified
    if memory_per_task and total_memory:
        max_by_memory = total_memory // memory_per_task
        if max_by_memory == 0:
            raise ValueError(f"Insufficient memory: need {memory_per_task}MB per task but only have {total_memory}MB")
        constraints['memory'] = max_by_memory
    
    # Find limiting constraint
    limiting_factor = min(constraints, key=constraints.get)
    batch_size = constraints[limiting_factor]
    
    log.info("Resource constraints:")
    for name, value in constraints.items():
        marker = "→" if name == limiting_factor else " "
        log.info(f"  {marker} {name}: {value} max parallel tasks")
    
    # Auto-calculate memory per task if not specified
    final_memory_per_task = memory_per_task
    if final_memory_per_task is None and total_memory:
        final_memory_per_task = total_memory // batch_size
        log.info(f"Auto-calculated memory per task: {final_memory_per_task} MB")
    
    return ResourceAllocation(
        batch_size=batch_size,
        cpus_per_task=cpus_per_task,
        memory_per_task=final_memory_per_task,
        limiting_factor=limiting_factor,
        total_cpus=available_cpus,
        total_memory=total_memory
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Calculate resources for 100 benchmarks
    allocation = calculate_resource_allocation(
        cpus_per_task=4,
        memory_per_task=2048,  # 2GB per task
        num_benchmarks=100
    )
    
    allocation.validate()
    allocation.log_summary()
    
    print(f"\nResult: Can run {allocation.batch_size} tasks in parallel")