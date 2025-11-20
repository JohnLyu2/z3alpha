#!/usr/bin/env python3
"""
SMT Competition Data Extractor

This script extracts benchmark results from SMT competition databases and
saves them in a CSV format with columns: benchmark, solver, score.

The score uses PAR2 (Penalized Average Runtime with factor 2) methodology:
- Solved benchmarks (sat/unsat): score = wallclock_time
- Unsolved/unknown/timeout: score = 2.0 * time_limit (penalty)
Lower scores are better (faster solving time).
"""

import sqlite3
import csv
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union


def get_evaluation_id(cursor: sqlite3.Cursor, year: int) -> Optional[int]:
    """Get the evaluation ID for a given year."""
    query = """
    SELECT id, name, date 
    FROM Evaluations 
    WHERE strftime('%Y', date) = ?
    ORDER BY date DESC
    LIMIT 1
    """
    cursor.execute(query, (str(year),))
    result = cursor.fetchone()
    
    if result:
        print(f"Found evaluation: {result[1]} (ID: {result[0]}, Date: {result[2]})")
        return result[0]
    else:
        print(f"No evaluation found for year {year}")
        return None


def calculate_par2_score(status: str, wallclock_time: Optional[float], 
                        time_limit: float = 20.0) -> float:
    """
    Calculate PAR2 (Penalized Average Runtime) score for a solver on a benchmark.
    
    PAR2 scoring (matching SMT-COMP methodology):
    - If solver solved (status = 'sat' or 'unsat') AND wallclockTime exists: 
      score = wallclockTime
    - Otherwise (unsolved/unknown/timeout/error): 
      score = 2.0 * time_limit (penalty)
    
    Lower scores are better (faster solving time).
    """
    if status in ('sat', 'unsat') and wallclock_time is not None:
        return wallclock_time
    else:
        return 2.0 * time_limit


def extract_results_for_logic(db_path: str, year: int, logic: str, 
                             eval_id: int, time_limit: float,
                             incremental: bool = False, output_dir: str = "output",
                             benchmark_base_path: str = "") -> None:
    """
    Extract results for a single logic type from SMT competition database and save to CSV.
    
    Args:
        db_path: Path to the SQLite database
        year: Year of the competition
        logic: Logic type (e.g., 'QF_BV', 'QF_LIA')
        eval_id: Evaluation ID for the year
        time_limit: Time limit for the evaluation
        incremental: Whether to include incremental benchmarks
        output_dir: Directory to save the CSV file
        benchmark_base_path: Base path to prepend to benchmark names
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"\nProcessing logic: {logic}")
    print("-" * 60)
    
    # Query to get results (matching the reference script's approach)
    query = """
    SELECT 
        b.name as benchmark_name,
        f.folderName as family_folder,
        b.logic,
        sv.fullName as solver_name,
        r.wallclockTime,
        r.status
    FROM Results r
    JOIN SolverVariants sv ON r.solverVariant = sv.id
    JOIN Queries q ON r.query = q.id
    JOIN Benchmarks b ON q.benchmark = b.id
    JOIN Families f ON b.family = f.id
    WHERE r.evaluation = ?
        AND b.logic = ?
        AND b.isIncremental = ?
    ORDER BY b.name, sv.fullName
    """
    
    cursor.execute(query, (eval_id, logic, 1 if incremental else 0))
    results = cursor.fetchall()
    
    if not results:
        print(f"  No results found for logic '{logic}'")
        conn.close()
        return
    
    print(f"  Found {len(results)} results")
    
    # Prepare CSV data
    csv_data = []
    for row in results:
        benchmark_name, family_folder, logic_str, solver_name, wallclock_time, status = row
        
        # Construct full benchmark path
        # Format: [LOGIC]/[DATE]-[BENCHMARKSET]/[FILENAME]
        full_benchmark_path = f"{logic_str}/{family_folder}/{benchmark_name}"
        if benchmark_base_path:
            full_benchmark_path = f"{benchmark_base_path}/{full_benchmark_path}"
        
        # Calculate PAR2 score
        par2_score = calculate_par2_score(status, wallclock_time, time_limit)
        
        csv_data.append({
            'benchmark': full_benchmark_path,
            'solver': solver_name,
            'score': par2_score
        })
    
    # Output filename
    incremental_suffix = "_incremental" if incremental else ""
    output_file = f"{output_dir}/{logic}_{year}{incremental_suffix}.csv"
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['benchmark', 'solver', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"  Successfully wrote {len(csv_data)} rows to {output_file}")
    
    # Print summary statistics
    solvers = set(row['solver'] for row in csv_data)
    benchmarks = set(row['benchmark'] for row in csv_data)
    print(f"  Unique solvers: {len(solvers)}")
    print(f"  Unique benchmarks: {len(benchmarks)}")
    
    conn.close()


def extract_results(db_path: str, year: int, logics: List[str], 
                   incremental: bool = False, output_dir: str = "output",
                   benchmark_base_path: str = "") -> None:
    """
    Extract results from SMT competition database and save to CSV files.
    Creates one CSV file per logic type.
    
    Args:
        db_path: Path to the SQLite database
        year: Year of the competition
        logics: List of logic types (e.g., ['QF_BV', 'QF_LIA'])
        incremental: Whether to include incremental benchmarks
        output_dir: Directory to save the CSV files
        benchmark_base_path: Base path to prepend to benchmark names
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Connect to database to get evaluation info
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get evaluation ID for the year
    eval_id = get_evaluation_id(cursor, year)
    if eval_id is None:
        print(f"Error: No evaluation found for year {year}")
        conn.close()
        sys.exit(1)
    
    # Get time limit for the evaluation
    cursor.execute("SELECT wallclockLimit FROM Evaluations WHERE id = ?", (eval_id,))
    result = cursor.fetchone()
    time_limit = result[0] if result and result[0] else 20.0
    print(f"Using time limit: {time_limit} seconds")
    
    conn.close()
    
    # Process each logic type
    print(f"\nProcessing {len(logics)} logic type(s)...")
    print("=" * 60)
    
    for logic in logics:
        extract_results_for_logic(
            db_path=db_path,
            year=year,
            logic=logic,
            eval_id=eval_id,
            time_limit=time_limit,
            incremental=incremental,
            output_dir=output_dir,
            benchmark_base_path=benchmark_base_path
        )
    
    print("\n" + "=" * 60)
    print(f"Completed processing all {len(logics)} logic type(s)")


def list_available_data(db_path: str) -> None:
    """List available evaluations and logics in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Available Evaluations:")
    print("-" * 80)
    cursor.execute("""
        SELECT id, name, date, wallclockLimit, memoryLimit 
        FROM Evaluations 
        ORDER BY date DESC
    """)
    for row in cursor.fetchall():
        print(f"  ID: {row[0]}, Name: {row[1]}, Date: {row[2]}, "
              f"Time Limit: {row[3]}s, Memory: {row[4]}GB")
    
    print("\nAvailable Logics:")
    print("-" * 80)
    cursor.execute("""
        SELECT DISTINCT b.logic, COUNT(*) as count
        FROM Benchmarks b
        GROUP BY b.logic
        ORDER BY b.logic
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} benchmarks")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract SMT competition results to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract results for a single logic from 2023 competition
  python smt_competition_extractor.py -d smtlib2025.sqlite -y 2023 -l QF_BV

  # Extract results for multiple logics
  python smt_competition_extractor.py -d smtlib2025.sqlite -y 2023 -l QF_BV QF_LIA QF_LRA

  # Extract incremental benchmarks for multiple logics
  python smt_competition_extractor.py -d smtlib2025.sqlite -y 2023 -l QF_BV QF_LIA --incremental

  # Specify custom benchmark base path and output directory
  python smt_competition_extractor.py -d smtlib2025.sqlite -y 2023 -l QF_BV QF_LIA \
      -b /home/user/benchmarks -o ./results

  # List available data
  python smt_competition_extractor.py -d smtlib2025.sqlite --list
        """
    )
    
    parser.add_argument('-d', '--database', required=True,
                       help='Path to SQLite database file')
    parser.add_argument('-y', '--year', type=int,
                       help='Year of the competition')
    parser.add_argument('-l', '--logics', nargs='+',
                       help='Logic type(s) (e.g., QF_BV, QF_LIA). Can specify multiple.')
    parser.add_argument('--incremental', action='store_true',
                       help='Include incremental benchmarks (default: False)')
    parser.add_argument('-o', '--output-dir', default='output',
                       help='Output directory for CSV file(s) (default: output)')
    parser.add_argument('-b', '--benchmark-base-path', default='',
                       help='Base path to prepend to benchmark paths')
    parser.add_argument('--list', action='store_true',
                       help='List available evaluations and logics')
    
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.database).exists():
        print(f"Error: Database file '{args.database}' not found")
        sys.exit(1)
    
    if args.list:
        list_available_data(args.database)
        return
    
    if not args.year or not args.logics:
        print("Error: --year and --logics are required (unless using --list)")
        parser.print_help()
        sys.exit(1)
    
    extract_results(
        db_path=args.database,
        year=args.year,
        logics=args.logics,
        incremental=args.incremental,
        output_dir=args.output_dir,
        benchmark_base_path=args.benchmark_base_path
    )


if __name__ == '__main__':
    main()