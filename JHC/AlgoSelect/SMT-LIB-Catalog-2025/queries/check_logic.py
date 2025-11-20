import sqlite3
import pandas as pd
import numpy as np

def analyze_instances_for_logic(db_path, logic_type, year=2024, incremental=False):
    """
    Analyze individual benchmark instances for a specific logic to understand performance gaps
    
    Args:
        db_path (str): Path to SQLite database
        logic_type (str): Logic to analyze (e.g., 'QF_SLIA')
        year (int): Year to analyze
        incremental (bool): Whether to include incremental benchmarks
    
    Returns:
        tuple: (instance_results_df, solver_summary_df, virtual_best_analysis_df)
    """
    
    conn = sqlite3.connect(db_path)
    
    # Get all results for this logic with PAR2 scores
    query = """
    SELECT 
        b.name AS benchmark_name,
        q.idx AS query_index,
        s.name AS solver_name,
        r.status,
        r.wallclockTime,
        e.wallclockLimit,
        CASE 
            WHEN r.status IN ('sat', 'unsat') AND r.wallclockTime IS NOT NULL 
            THEN r.wallclockTime
            ELSE 2.0 * e.wallclockLimit
        END AS par2_score,
        e.name AS evaluation_name
    FROM Results r
    JOIN SolverVariants sv ON r.solverVariant = sv.id
    JOIN Solvers s ON sv.solver = s.id
    JOIN Queries q ON r.query = q.id
    JOIN Benchmarks b ON q.benchmark = b.id
    JOIN Evaluations e ON r.evaluation = e.id
    WHERE b.logic = ?
    AND strftime('%Y', e.date) = ?
    AND b.isIncremental = ?
    ORDER BY b.name, q.idx, s.name
    """
    
    instance_results = pd.read_sql_query(query, conn, params=[logic_type, str(year), incremental])
    
    if instance_results.empty:
        print(f"No results found for {logic_type} in {year}")
        conn.close()
        return None, None, None
    
    print(f"Found {len(instance_results)} individual solver results for {logic_type}")
    print(f"Covering {instance_results['benchmark_name'].nunique()} unique benchmarks")
    print(f"With {instance_results['solver_name'].nunique()} different solvers")
    
    # Create instance identifier
    instance_results['instance_id'] = instance_results['benchmark_name'] + '_' + instance_results['query_index'].astype(str)
    
    # Analyze solver performance summary
    solver_summary = instance_results.groupby('solver_name').agg({
        'par2_score': ['count', 'mean', 'median', 'min', 'max'],
        'status': lambda x: (x.isin(['sat', 'unsat'])).sum(),
        'wallclockTime': lambda x: x.notna().sum()
    }).round(3)
    
    solver_summary.columns = ['total_instances', 'avg_par2', 'median_par2', 'min_par2', 'max_par2', 'solved_instances', 'recorded_times']
    solver_summary['solve_rate_%'] = (solver_summary['solved_instances'] / solver_summary['total_instances'] * 100).round(2)
    solver_summary = solver_summary.sort_values('avg_par2')
    
    # Find virtual best performance per instance
    virtual_best_per_instance = instance_results.groupby('instance_id').agg({
        'par2_score': 'min',
        'solver_name': lambda x: x.loc[instance_results.loc[x.index, 'par2_score'].idxmin()],
        'status': lambda x: x.loc[instance_results.loc[x.index, 'par2_score'].idxmin()],
        'wallclockTime': lambda x: x.loc[instance_results.loc[x.index, 'par2_score'].idxmin()]
    })
    virtual_best_per_instance.columns = ['virtual_best_par2', 'best_solver', 'best_status', 'best_time']
    
    # Add single best solver performance for comparison
    best_solver = solver_summary.index[0]  # Best solver by average PAR2
    best_solver_results = instance_results[instance_results['solver_name'] == best_solver].set_index('instance_id')
    
    comparison = virtual_best_per_instance.join(
        best_solver_results[['par2_score', 'status', 'wallclockTime']].rename(columns={
            'par2_score': 'single_best_par2',
            'status': 'single_best_status', 
            'wallclockTime': 'single_best_time'
        }), 
        how='inner'
    )
    
    # Calculate gap for each instance
    comparison['gap_absolute'] = comparison['single_best_par2'] - comparison['virtual_best_par2']
    comparison['gap_percent'] = ((comparison['single_best_par2'] - comparison['virtual_best_par2']) / comparison['virtual_best_par2'] * 100).round(2)
    
    # Sort by largest gaps
    comparison = comparison.sort_values('gap_percent', ascending=False)
    
    conn.close()
    
    return instance_results, solver_summary, comparison

def analyze_problematic_instances(comparison_df, top_n=10):
    """
    Analyze the most problematic instances (largest performance gaps)
    """
    print(f"\nTOP {top_n} INSTANCES WITH LARGEST PERFORMANCE GAPS:")
    print("=" * 80)
    
    top_gaps = comparison_df.head(top_n)
    
    for i, (instance_id, row) in enumerate(top_gaps.iterrows(), 1):
        print(f"\n{i}. Instance: {instance_id}")
        print(f"   Gap: {row['gap_percent']:.1f}% ({row['gap_absolute']:.2f}s difference)")
        print(f"   Single Best: {row['single_best_par2']:.2f}s ({row['single_best_status']})")
        print(f"   Virtual Best: {row['virtual_best_par2']:.2f}s by {row['best_solver']} ({row['best_status']})")
    
    return top_gaps

def summarize_gap_analysis(comparison_df, solver_summary_df):
    """
    Provide summary statistics about the performance gap
    """
    print(f"\nPERFORMANCE GAP SUMMARY:")
    print("=" * 50)
    
    total_instances = len(comparison_df)
    avg_gap = comparison_df['gap_percent'].mean()
    median_gap = comparison_df['gap_percent'].median()
    max_gap = comparison_df['gap_percent'].max()
    
    print(f"Total instances analyzed: {total_instances}")
    print(f"Average performance gap: {avg_gap:.1f}%")
    print(f"Median performance gap: {median_gap:.1f}%")
    print(f"Maximum performance gap: {max_gap:.1f}%")
    
    # Count how many instances have significant gaps
    large_gaps = (comparison_df['gap_percent'] > 100).sum()
    huge_gaps = (comparison_df['gap_percent'] > 1000).sum()
    
    print(f"Instances with >100% gap: {large_gaps} ({large_gaps/total_instances*100:.1f}%)")
    print(f"Instances with >1000% gap: {huge_gaps} ({huge_gaps/total_instances*100:.1f}%)")
    
    # Show which solvers contribute most to virtual best
    print(f"\nVIRTUAL BEST SOLVER CONTRIBUTORS:")
    print("-" * 40)
    best_solver_counts = comparison_df['best_solver'].value_counts()
    for solver, count in best_solver_counts.head(5).items():
        percentage = count / total_instances * 100
        print(f"{solver}: {count} instances ({percentage:.1f}%)")

# Usage example
if __name__ == "__main__":
    db_path = "smtlib2025.sqlite"
    
    # Analyze QF_SLIA in detail
    print("Analyzing QF_SLIA performance in detail...")
    instance_results, solver_summary, comparison = analyze_instances_for_logic(
        db_path, "QF_SLIA", year=2024, incremental=False
    )
    
    if instance_results is not None:
        print(f"\nSOLVER PERFORMANCE SUMMARY FOR QF_SLIA:")
        print("=" * 60)
        print(solver_summary)
        
        # Analyze most problematic instances
        top_problems = analyze_problematic_instances(comparison)
        
        # Overall gap analysis
        summarize_gap_analysis(comparison, solver_summary)
        
        # Save detailed results
        comparison.to_csv('qf_slia_instance_analysis.csv')
        solver_summary.to_csv('qf_slia_solver_summary.csv')
        
        print(f"\nDetailed results saved to:")
        print("- qf_slia_instance_analysis.csv (per-instance comparison)")
        print("- qf_slia_solver_summary.csv (solver summary)")
        
        # Show some statistics about Z3-Noodler specifically
        if 'Z3-Noodler' in solver_summary.index:
            z3n_stats = solver_summary.loc['Z3-Noodler']
            print(f"\nZ3-NOODLER SPECIFIC STATS:")
            print(f"- Average PAR2: {z3n_stats['avg_par2']}")
            print(f"- Solve rate: {z3n_stats['solve_rate_%']}%")
            print(f"- Solved {z3n_stats['solved_instances']} out of {z3n_stats['total_instances']} instances")
            
            # How often is Z3-Noodler the virtual best?
            z3n_best_count = (comparison['best_solver'] == 'Z3-Noodler').sum()
            z3n_best_pct = z3n_best_count / len(comparison) * 100
            print(f"- Z3-Noodler is virtual best on {z3n_best_count} instances ({z3n_best_pct:.1f}%)")