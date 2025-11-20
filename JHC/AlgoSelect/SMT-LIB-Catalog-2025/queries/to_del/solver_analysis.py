import sqlite3
import pandas as pd
import numpy as np

def analyze_solver_performance(db_path, solver_name, logic_type, year=None):
    """
    Calculate performance metrics for a specific solver on a specific logic type
    
    Args:
        db_path (str): Path to the SQLite database file
        solver_name (str): Name of the solver to analyze
        logic_type (str): Logic type to filter by (e.g., 'QF_LIA', 'QF_ABV')
        year (int, optional): Year to filter by. If None, analyzes all years
    
    Returns:
        dict: Performance metrics including solve rate, times, status distribution
    """
    
    # Build query with optional year filter
    year_filter = "AND strftime('%Y', e.date) = ?" if year else ""
    params = [solver_name, logic_type]
    if year:
        params.append(str(year))
    
    query = f"""
    SELECT 
        s.name AS solver_name,
        b.logic,
        r.cpuTime,
        r.wallclockTime,
        r.status,
        b.name AS instance_name,
        e.name AS evaluation_name,
        e.date AS evaluation_date
    FROM Results r
    JOIN SolverVariants sv ON r.solverVariant = sv.id
    JOIN Solvers s ON sv.solver = s.id
    JOIN Queries q ON r.query = q.id
    JOIN Benchmarks b ON q.benchmark = b.id
    JOIN Evaluations e ON r.evaluation = e.id
    WHERE s.name = ? 
    AND b.logic = ?
    {year_filter}
    ORDER BY b.name;
    """
    
    # Execute query
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return {
            'error': f"No results found for solver '{solver_name}' on logic '{logic_type}'"
        }
    
    # Calculate performance metrics
    total_instances = len(df)
    
    # Status analysis
    status_counts = df['status'].value_counts()
    solved_instances = len(df[df['status'].isin(['sat', 'unsat'])])
    unknown_instances = len(df[df['status'] == 'unknown'])
    
    solve_rate = (solved_instances / total_instances) * 100
    
    # Time analysis (excluding None values)
    cpu_times = df['cpuTime'].dropna()
    wall_times = df['wallclockTime'].dropna()
    
    # Performance metrics
    performance = {
        'solver_name': solver_name,
        'logic_type': logic_type,
        'year': year if year else 'All years',
        'total_instances': total_instances,
        'solved_instances': solved_instances,
        'unknown_instances': unknown_instances,
        'solve_rate_percent': round(solve_rate, 2),
        'status_distribution': status_counts.to_dict(),
        
        # CPU Time metrics
        'cpu_time_stats': {
            'count': len(cpu_times),
            'mean': round(cpu_times.mean(), 3) if not cpu_times.empty else None,
            'median': round(cpu_times.median(), 3) if not cpu_times.empty else None,
            'min': round(cpu_times.min(), 3) if not cpu_times.empty else None,
            'max': round(cpu_times.max(), 3) if not cpu_times.empty else None,
            'std': round(cpu_times.std(), 3) if not cpu_times.empty else None
        },
        
        # Wall Clock Time metrics
        'wallclock_time_stats': {
            'count': len(wall_times),
            'mean': round(wall_times.mean(), 3) if not wall_times.empty else None,
            'median': round(wall_times.median(), 3) if not wall_times.empty else None,
            'min': round(wall_times.min(), 3) if not wall_times.empty else None,
            'max': round(wall_times.max(), 3) if not wall_times.empty else None,
            'std': round(wall_times.std(), 3) if not wall_times.empty else None
        }
    }
    
    return performance

def compare_solvers_on_logic(db_path, solver_names, logic_type, year=None):
    """
    Compare multiple solvers on the same logic type
    
    Args:
        db_path (str): Path to the SQLite database file
        solver_names (list): List of solver names to compare
        logic_type (str): Logic type to analyze
        year (int, optional): Year to filter by
    
    Returns:
        pandas.DataFrame: Comparison table of solver performances
    """
    
    comparisons = []
    
    for solver in solver_names:
        perf = analyze_solver_performance(db_path, solver, logic_type, year)
        
        if 'error' not in perf:
            comparisons.append({
                'Solver': perf['solver_name'],
                'Total_Instances': perf['total_instances'],
                'Solved_Instances': perf['solved_instances'],
                'Solve_Rate_%': perf['solve_rate_percent'],
                'Avg_CPU_Time': perf['cpu_time_stats']['mean'],
                'Median_CPU_Time': perf['cpu_time_stats']['median'],
                'Avg_Wall_Time': perf['wallclock_time_stats']['mean'],
                'Median_Wall_Time': perf['wallclock_time_stats']['median']
            })
    
    if not comparisons:
        print("No valid results found for any solver")
        return None
    
    df = pd.DataFrame(comparisons)
    return df.sort_values('Solve_Rate_%', ascending=False)

def get_available_logics(db_path, year=None):
    """
    Get list of available logic types in the database
    
    Args:
        db_path (str): Path to database
        year (int, optional): Filter by year
    
    Returns:
        pandas.DataFrame: Available logics with instance counts
    """
    
    year_filter = "WHERE strftime('%Y', e.date) = ?" if year else ""
    params = [str(year)] if year else []
    
    query = f"""
    SELECT 
        b.logic,
        COUNT(*) as instance_count,
        COUNT(DISTINCT s.name) as solver_count
    FROM Results r
    JOIN SolverVariants sv ON r.solverVariant = sv.id
    JOIN Solvers s ON sv.solver = s.id
    JOIN Queries q ON r.query = q.id
    JOIN Benchmarks b ON q.benchmark = b.id
    JOIN Evaluations e ON r.evaluation = e.id
    {year_filter}
    GROUP BY b.logic
    ORDER BY instance_count DESC;
    """
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df

def get_available_solvers(db_path, logic_type=None, year=None):
    """
    Get list of available solvers, optionally filtered by logic and year
    
    Args:
        db_path (str): Path to database
        logic_type (str, optional): Filter by logic type
        year (int, optional): Filter by year
    
    Returns:
        pandas.DataFrame: Available solvers
    """
    
    filters = []
    params = []
    
    if logic_type:
        filters.append("b.logic = ?")
        params.append(logic_type)
    
    if year:
        filters.append("strftime('%Y', e.date) = ?")
        params.append(str(year))
    
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    
    query = f"""
    SELECT 
        s.name as solver_name,
        COUNT(*) as total_results,
        COUNT(DISTINCT b.logic) as logic_count
    FROM Results r
    JOIN SolverVariants sv ON r.solverVariant = sv.id
    JOIN Solvers s ON sv.solver = s.id
    JOIN Queries q ON r.query = q.id
    JOIN Benchmarks b ON q.benchmark = b.id
    JOIN Evaluations e ON r.evaluation = e.id
    {where_clause}
    GROUP BY s.name
    ORDER BY total_results DESC;
    """
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df

def print_performance_report(performance):
    """
    Pretty print performance metrics
    
    Args:
        performance (dict): Performance metrics from analyze_solver_performance
    """
    
    if 'error' in performance:
        print(f"Error: {performance['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE REPORT")
    print(f"{'='*60}")
    print(f"Solver: {performance['solver_name']}")
    print(f"Logic: {performance['logic_type']}")
    print(f"Year: {performance['year']}")
    print(f"{'='*60}")
    
    print(f"\nSOLVE STATISTICS:")
    print(f"  Total instances: {performance['total_instances']}")
    print(f"  Solved instances: {performance['solved_instances']}")
    print(f"  Unknown instances: {performance['unknown_instances']}")
    print(f"  Solve rate: {performance['solve_rate_percent']}%")
    
    print(f"\nSTATUS DISTRIBUTION:")
    for status, count in performance['status_distribution'].items():
        print(f"  {status}: {count}")
    
    cpu_stats = performance['cpu_time_stats']
    if cpu_stats['mean'] is not None:
        print(f"\nCPU TIME STATISTICS (seconds):")
        print(f"  Count: {cpu_stats['count']}")
        print(f"  Mean: {cpu_stats['mean']}")
        print(f"  Median: {cpu_stats['median']}")
        print(f"  Min: {cpu_stats['min']}")
        print(f"  Max: {cpu_stats['max']}")
        print(f"  Std Dev: {cpu_stats['std']}")
    
    wall_stats = performance['wallclock_time_stats']
    if wall_stats['mean'] is not None:
        print(f"\nWALL CLOCK TIME STATISTICS (seconds):")
        print(f"  Count: {wall_stats['count']}")
        print(f"  Mean: {wall_stats['mean']}")
        print(f"  Median: {wall_stats['median']}")
        print(f"  Min: {wall_stats['min']}")
        print(f"  Max: {wall_stats['max']}")
        print(f"  Std Dev: {wall_stats['std']}")

# Example usage
if __name__ == "__main__":
    db_path = "smtlib2025.sqlite"
    
    # Example 1: Analyze single solver performance
    print("Analyzing Z3 performance on QF_LIA logic...")
    perf = analyze_solver_performance(db_path, "Z3", "QF_LIA", 2024)
    print_performance_report(perf)
    
    # Example 2: Compare multiple solvers
    print("\nComparing solvers on QF_ABV logic...")
    solvers = ["Z3", "CVC4", "Yices"]
    comparison = compare_solvers_on_logic(db_path, solvers, "QF_ABV", 2024)
    if comparison is not None:
        print(comparison)
    
    # Example 3: See available logics
    print("\nAvailable logics in 2024:")
    logics = get_available_logics(db_path, 2024)
    print(logics.head(10))
    
    # Example 4: See available solvers for a specific logic
    print("\nSolvers that worked on QF_LIA in 2024:")
    solvers_df = get_available_solvers(db_path, "QF_LIA", 2024)
    print(solvers_df.head(10))