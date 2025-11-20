import sqlite3
import pandas as pd

def get_smt_results(db_path, year):
    """
    Retrieve SMT competition results for a specific year
    
    Args:
        db_path (str): Path to the SQLite database file
        year (int): Year to filter results by (e.g., 2024)
    
    Returns:
        pandas.DataFrame: Results with columns: Solver_name, cpuTime, wallclockTime, smt_instance
    """
    
    query = """
    SELECT 
        s.name AS Solver_name,
        r.cpuTime,
        r.wallclockTime,
        b.name AS smt_instance,
        b.logic,
        r.status
    FROM Results r
    JOIN SolverVariants sv ON r.solverVariant = sv.id
    JOIN Solvers s ON sv.solver = s.id
    JOIN Queries q ON r.query = q.id
    JOIN Benchmarks b ON q.benchmark = b.id
    JOIN Evaluations e ON r.evaluation = e.id
    WHERE strftime('%Y', e.date) = ?
    ORDER BY s.name, b.name;
    """
    
    # Connect and execute query
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=(str(year),))
    conn.close()
    
    print(f"Retrieved {len(df)} results for year {year}")
    return df

# Usage example
if __name__ == "__main__":
    # Set your database path and target year
    db_path = "smtlib2025.sqlite"
    target_year = 2025
    
    # Get the results
    results = get_smt_results(db_path, target_year)
    
    # Save to CSV
    output_file = f"smt_results_{target_year}.csv"
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Preview first few rows
    print("\nFirst 5 results:")
    print(results.head())