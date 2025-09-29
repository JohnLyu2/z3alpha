import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """Container for system resource information"""
    total_cpus: int
    available_cpus: int
    total_memory_mb: Optional[int]
    available_memory_mb: Optional[int]
    slurm_cpus: Optional[int]
    slurm_memory_mb: Optional[int]
    is_slurm: bool


@dataclass
class TaskAllocation:
    """Container for task allocation information"""
    batch_size: int
    cpus_per_task: int
    memory_per_task_mb: Optional[int]
    total_cpus_used: int
    total_memory_used_mb: Optional[int]
    limiting_factor: str
    spare_cpus: int
    spare_memory_mb: Optional[int]


def log_system_resources(resources: SystemResources):
    """
    Log system resource information in a clean, consolidated format.
    Replaces ~10 scattered log statements.
    """
    log.info("=" * 60)
    log.info("SYSTEM RESOURCES")
    log.info("=" * 60)
    
    # CPU Information
    log.info(f"CPUs:")
    log.info(f"  Total (system):     {resources.total_cpus}")
    if resources.is_slurm:
        log.info(f"  Allocated (Slurm):  {resources.slurm_cpus}")
        log.info(f"  Available:          {resources.available_cpus} ← using this")
    else:
        log.info(f"  Available:          {resources.available_cpus}")
    
    # Memory Information
    if resources.total_memory_mb or resources.slurm_memory_mb:
        log.info(f"Memory:")
        if resources.total_memory_mb:
            log.info(f"  Total (system):     {resources.total_memory_mb:,} MB")
        if resources.is_slurm and resources.slurm_memory_mb:
            log.info(f"  Allocated (Slurm):  {resources.slurm_memory_mb:,} MB")
            log.info(f"  Available:          {resources.available_memory_mb:,} MB ← using this")
        elif resources.available_memory_mb:
            log.info(f"  Available:          {resources.available_memory_mb:,} MB")
    
    # Slurm status
    if resources.is_slurm:
        import os
        job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        log.info(f"Running in Slurm job: {job_id}")
    
    log.info("=" * 60)


def log_task_allocation(allocation: TaskAllocation, num_benchmarks: int):
    """
    Log task allocation information in a clean, consolidated format.
    Replaces ~15 scattered log statements.
    """
    log.info("=" * 60)
    log.info("TASK ALLOCATION")
    log.info("=" * 60)
    
    # Basic allocation
    log.info(f"Configuration:")
    log.info(f"  Benchmarks:         {num_benchmarks}")
    log.info(f"  Parallel tasks:     {allocation.batch_size}")
    log.info(f"  CPUs per task:      {allocation.cpus_per_task}")
    if allocation.memory_per_task_mb:
        log.info(f"  Memory per task:    {allocation.memory_per_task_mb:,} MB")
    
    # Resource usage
    log.info(f"Resource Usage:")
    cpu_efficiency = (allocation.total_cpus_used / (allocation.total_cpus_used + allocation.spare_cpus)) * 100
    log.info(f"  Total CPUs used:    {allocation.total_cpus_used} (efficiency: {cpu_efficiency:.1f}%)")
    
    if allocation.memory_per_task_mb and allocation.total_memory_used_mb:
        if allocation.spare_memory_mb is not None:
            total_mem = allocation.total_memory_used_mb + allocation.spare_memory_mb
            mem_efficiency = (allocation.total_memory_used_mb / total_mem) * 100
            log.info(f"  Total memory used:  {allocation.total_memory_used_mb:,} MB (efficiency: {mem_efficiency:.1f}%)")
        else:
            log.info(f"  Total memory used:  {allocation.total_memory_used_mb:,} MB")
    
    # Constraints and limiting factors
    log.info(f"Constraints:")
    log.info(f"  Limiting factor:    {allocation.limiting_factor}")
    if allocation.spare_cpus > 0:
        log.info(f"  Spare CPUs:         {allocation.spare_cpus}")
    if allocation.spare_memory_mb and allocation.spare_memory_mb > 0:
        log.info(f"  Spare memory:       {allocation.spare_memory_mb:,} MB")
    
    log.info("=" * 60)


def log_validation_result(passed: bool, cpu_ok: bool, memory_ok: bool, 
                          used_cpus: int, total_cpus: int,
                          used_memory: Optional[int], total_memory: Optional[int]):
    """
    Log resource validation results in a consolidated format.
    Replaces the validate() method logging.
    """
    log.info("=" * 60)
    log.info("RESOURCE VALIDATION")
    log.info("=" * 60)
    
    # CPU validation
    if cpu_ok:
        cpu_efficiency = (used_cpus / total_cpus) * 100
        log.info(f"✅ CPU allocation OK: {used_cpus}/{total_cpus} ({cpu_efficiency:.1f}%)")
    else:
        log.error(f"❌ CPU VIOLATION: Using {used_cpus} but only {total_cpus} available")
    
    # Memory validation
    if used_memory and total_memory:
        if memory_ok:
            mem_efficiency = (used_memory / total_memory) * 100
            log.info(f"✅ Memory allocation OK: {used_memory:,}/{total_memory:,} MB ({mem_efficiency:.1f}%)")
        else:
            log.error(f"❌ MEMORY VIOLATION: Using {used_memory:,} MB but only {total_memory:,} MB available")
    elif used_memory:
        log.info(f"✅ Memory per task: {used_memory:,} MB (total unknown)")
    
    log.info("=" * 60)
    
    if not passed:
        raise ValueError("Resource allocation exceeds available resources")


def log_execution_summary(num_benchmarks: int, batch_size: int, 
                         cpus_per_task: int, memory_per_task: Optional[int]):
    """
    Log execution start summary.
    Replaces scattered log statements at execution start.
    """
    log.info("=" * 60)
    log.info("STARTING PARALLEL EXECUTION")
    log.info("=" * 60)
    log.info(f"Benchmarks:       {num_benchmarks}")
    log.info(f"Parallel tasks:   {batch_size}")
    log.info(f"CPUs per task:    {cpus_per_task}")
    if memory_per_task:
        log.info(f"Memory per task:  {memory_per_task:,} MB")
    log.info("=" * 60)


def log_batch_progress(batch_num: int, start_idx: int, end_idx: int, 
                      duration: Optional[float] = None):
    """
    Log batch progress in a clean format.
    """
    if duration:
        log.info(f"Batch {batch_num}: tasks {start_idx}-{end_idx-1} completed in {duration:.2f}s")
    else:
        log.info(f"Batch {batch_num}: processing tasks {start_idx}-{end_idx-1}")


def log_evaluation_results(total_benchmarks: int, solved: int, 
                          par2: float, par10: float, total_time: float):
    """
    Log final evaluation results in a consolidated format.
    Replaces scattered result logging.
    """
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"Benchmarks:       {total_benchmarks}")
    log.info(f"Solved:           {solved} ({solved/total_benchmarks*100:.1f}%)")
    log.info(f"Unsolved:         {total_benchmarks - solved}")
    log.info(f"PAR2 score:       {par2:.2f}")
    log.info(f"PAR10 score:      {par10:.2f}")
    log.info(f"Total time:       {total_time:.2f}s")
    log.info(f"Avg per benchmark: {total_time/total_benchmarks:.2f}s")
    log.info("=" * 60)
