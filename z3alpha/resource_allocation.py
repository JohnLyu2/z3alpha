import os
import logging
import psutil
from z3alpha.resource_logging import log_system_resources, SystemResources, TaskAllocation, log_task_allocation, log_validation_result

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
        """Validate using consolidated logging"""
        cpu_ok = self.total_cpus_used <= self.total_cpus
        memory_ok = True
        if self.memory_per_task and self.total_memory:
            memory_ok = self.total_memory_used <= self.total_memory
        
        passed = cpu_ok and memory_ok
        
        log_validation_result(
            passed=passed,
            cpu_ok=cpu_ok,
            memory_ok=memory_ok,
            used_cpus=self.total_cpus_used,
            total_cpus=self.total_cpus,
            used_memory=self.total_memory_used,
            total_memory=self.total_memory
        )

    def log_summary(self, num_benchmarks: int):
        """Log summary using consolidated logging"""
        allocation_info = TaskAllocation(
            batch_size=self.batch_size,
            cpus_per_task=self.cpus_per_task,
            memory_per_task_mb=self.memory_per_task,
            total_cpus_used=self.total_cpus_used,
            total_memory_used_mb=self.total_memory_used,
            limiting_factor=self.limiting_factor,
            spare_cpus=self.spare_cpus,
            spare_memory_mb=self.spare_memory
        )
        
        log_task_allocation(allocation_info, num_benchmarks)


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
    # Gather all resource information
    slurm_cpus, slurm_mem, is_slurm = get_slurm_resources()
    total_cpus = os.cpu_count()
    available_cpus = slurm_cpus if slurm_cpus else total_cpus
    
    # Get memory info
    total_memory = None
    if slurm_mem:
        total_memory = slurm_mem
    else:
        try:
            total_memory = psutil.virtual_memory().total // (1024 * 1024)
        except:
            pass
    
    # Log system resources in one consolidated call
    system_resources = SystemResources(
        total_cpus=total_cpus,
        available_cpus=available_cpus,
        total_memory_mb=total_memory if not slurm_mem else None,
        available_memory_mb=total_memory,
        slurm_cpus=slurm_cpus,
        slurm_memory_mb=slurm_mem,
        is_slurm=is_slurm
    )
    log_system_resources(system_resources)
    
    # Calculate constraints (no logging here)
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