import sqlite3
import pandas as pd
import numpy as np
import math

class StdevFunc:
    """Calculates sample standard deviation using Welford's online algorithm."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0 # Sum of squares of differences from the current mean

    def step(self, value):
        if value is None:
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def finalize(self):
        if self.count < 2:
            return None # Std dev is undefined for 0 or 1 values
        
        variance = self.m2 / (self.count - 1) # Sample variance
        return math.sqrt(variance)
    
class SolverRanking:
    def __init__(self, db_path, debug=False):
        self.db_path = db_path
        self.conn = None
        self.debug = debug 
        self.debug_limit = 100  # Limit for debug output
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.create_aggregate("stdev", 1, StdevFunc)
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def create_logic_description_stats_table(self):
        """
        NEW: Create an auxiliary table with description length stats per logic.
        """
        print("Step 1.5: Creating logic description stats table...")
        
        query = """
        CREATE TEMP TABLE logic_description_stats AS
        SELECT
            b.logic,
            MAX(LENGTH(b.description)) AS max_desc_length,
            AVG(LENGTH(b.description)) AS avg_desc_length,
            stdev(LENGTH(b.description)) AS stdev_desc_length,
            MIN(LENGTH(b.description)) AS min_desc_length
        FROM Benchmarks b
        WHERE b.description IS NOT NULL AND LENGTH(b.description) > 0
        GROUP BY b.logic;
        """
        
        self.conn.execute(query)
        self.conn.commit()
        
        count = self.conn.execute("SELECT COUNT(*) FROM logic_description_stats").fetchone()[0]
        print(f"   Created description stats table for {count} logics")
        return count
    
    def create_par2_results_table(self, year=None, incremental=None):
        """
        Step 1: Create auxiliary table with PAR2 scores for each result
        PAR2 = wallclockTime if solved, 2 * wallclockLimit if unsolved/unknown
        
        Args:
            year (int, optional): Filter by evaluation year
            incremental (bool, optional): 
                True = only incremental benchmarks
                False = only non-incremental benchmarks  
                None = include both types
        """
        print("Step 1: Creating PAR2 results table...")
        
        filters = []
        params = []
        
        if year:
            filters.append("strftime('%Y', e.date) = ?")
            params.append(str(year))
        
        if incremental is not None:
            filters.append("b.isIncremental = ?")
            params.append(incremental)
        
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""
        
        query = f"""
        CREATE TEMP TABLE par2_results AS
        SELECT 
            r.id as result_id,
            s.name AS solver_name,
            b.logic,
            b.name AS benchmark_name,
            b.isIncremental,
            q.id AS query_id,
            r.status,
            r.wallclockTime,
            e.wallclockLimit,
            e.name AS evaluation_name,
            e.date AS evaluation_date,
            CASE 
                WHEN r.status IN ('sat', 'unsat') AND r.wallclockTime IS NOT NULL 
                THEN r.wallclockTime
                ELSE 2.0 * e.wallclockLimit
            END AS par2_score
        FROM Results r
        JOIN SolverVariants sv ON r.solverVariant = sv.id
        JOIN Solvers s ON sv.solver = s.id
        JOIN Queries q ON r.query = q.id
        JOIN Benchmarks b ON q.benchmark = b.id
        JOIN Evaluations e ON r.evaluation = e.id
        {where_clause}
        """
        
        self.conn.execute(query, params)
        self.conn.commit()  # Ensure the table is committed before counting
        
        # Get count for verification
        count_query = "SELECT COUNT(*) FROM par2_results"
        count = self.conn.execute(count_query).fetchone()[0]
        print(f"   Created PAR2 results table with {count} entries")
        
        if self.debug: 
            self._debug_par2_results(logic_to_check="QF_ABVFPLRA")
            self._debug_full_logic_details(logic_to_check='QF_ABVFPLRA')
        
        return count
    
    
    def _debug_full_logic_details(self, logic_to_check='QF_ABVFPLRA', evaluation_name='SMT-COMP 2024'):
        """
        Prints all raw results for a specific logic and evaluation, sorted by benchmark and solver,
        to allow for manual verification of SBS and VBS.
        """
        print(f"\n--- DEBUG: Raw Results for Logic '{logic_to_check}' from Evaluation '{evaluation_name}' ---")
        
        query = """
        SELECT
            benchmark_name,
            solver_name,
            status,
            ROUND(par2_score, 4) as par2_score, 
            evaluation_name,
            evaluation_date
        FROM
            par2_results
        WHERE
            logic = ?
            AND evaluation_name = ?  -- This is the new filter
        ORDER BY
            benchmark_name, solver_name;
        """
        
        try:
            # Pass both parameters to the query
            details_df = pd.read_sql_query(query, self.conn, params=[logic_to_check, evaluation_name])
            
            if not details_df.empty:
                print("This table contains all the data needed to manually verify SBS and VBS.")
                print(details_df.to_string())
            else:
                print(f"No results found for logic = '{logic_to_check}' and evaluation = '{evaluation_name}'.")
                
        except Exception as e:
            print(f"An error occurred during the debug query: {e}")

        print("--- END: DEBUG ---\n")
    
    
    def _debug_par2_results(self, logic_to_check="QF_SLIA"):
        """Helper function to debug PAR2 results table"""
        print(f"\n--- DEBUG: Inspecting 'par2_results' for logic '{logic_to_check}' ---")
    
        debug_query = """
        SELECT solver_name, status, wallclockTime, wallclockLimit, par2_score
        FROM par2_results
        WHERE logic = ?
        LIMIT ?;
        """
    
        try:
            debug_df = pd.read_sql_query(debug_query, self.conn, params=[logic_to_check, self.debug_limit])
            
            if not debug_df.empty:
                print(debug_df.to_string())
            else:
                print(f"No data found for logic = '{logic_to_check}' with the current filters.")
                
        except Exception as e:
            print(f"An error occurred during debugging query: {e}")
            
        print("--- END: DEBUG --- \n")

        
    def create_solver_logic_performance_table(self):
        """
        Step 2: Calculate average PAR2 score for each solver on each logic
        """
        print("Step 2: Creating solver-logic performance table...")
        
        query = """
        CREATE TEMP TABLE solver_logic_performance AS
        SELECT 
            solver_name,
            logic,
            COUNT(*) as total_instances,
            COUNT(CASE WHEN status IN ('sat', 'unsat') THEN 1 END) as solved_instances,
            AVG(par2_score) as avg_par2_score,
            
            -- ADD THIS LINE to calculate the sum of time for solved instances
            SUM(CASE WHEN status IN ('sat', 'unsat') THEN wallclockTime ELSE 0 END) as total_solved_time,

            evaluation_name,
            evaluation_date
        FROM par2_results
        GROUP BY solver_name, logic, evaluation_name, evaluation_date
        """
        
        self.conn.execute(query)
        self.conn.commit()
        
        # Get count for verification
        count_query = "SELECT COUNT(*) FROM solver_logic_performance"
        count = self.conn.execute(count_query).fetchone()[0]
        print(f"   Created solver-logic performance table with {count} entries")
        
        if self.debug:
            # Call the new leaderboard function
            self._debug_logic_leaderboard(logic_to_check='QF_ABVFPLRA')
                    
        return count
    
    def _debug_logic_leaderboard(self, logic_to_check='QF_SLIA'):
        """
        Prints the top 4 performing solvers for a given logic,
        ranked by the number of solved instances.
        """
        print(f"\n--- DEBUG: Top 4 solvers for logic '{logic_to_check}' (by solved instances) ---")
        
        query = """
        SELECT
            solver_name,
            solved_instances,
            total_instances,
            ROUND(avg_par2_score, 2) as avg_par2_score,
            
            -- ADD THIS LINE to display the new time column
            ROUND(total_solved_time, 2) as total_solved_time

        FROM
            solver_logic_performance
        WHERE
            logic = ?
        ORDER BY
            solved_instances DESC
        LIMIT 4;
        """
        
        try:
            leaderboard_df = pd.read_sql_query(query, self.conn, params=[logic_to_check])
            
            if not leaderboard_df.empty:
                print(leaderboard_df.to_string(index=False))
            else:
                print(f"No performance data found for logic = '{logic_to_check}'.")
                
        except Exception as e:
            print(f"An error occurred during the debug query: {e}")

        print("--- END: DEBUG ---\n")
    
    def create_single_best_solver_table(self):
        """
        Step 3: Find the single best solver for each logic type
        """
        print("Step 3: Creating single best solver table...")
        
        query = """
        CREATE TEMP TABLE single_best_solver AS
        SELECT 
            logic,
            solver_name as best_solver,
            avg_par2_score as best_par2_score,
            total_instances,
            solved_instances,
            ROUND(100.0 * solved_instances / total_instances, 2) as solve_rate_percent,
            evaluation_name,
            evaluation_date,
            ROW_NUMBER() OVER (PARTITION BY logic ORDER BY avg_par2_score ASC) as rank
        FROM solver_logic_performance
        """
        
        self.conn.execute(query)
        self.conn.commit()
        
        # Filter to get only the best solver per logic
        query_best = """
        CREATE TEMP TABLE single_best_final AS
        SELECT 
            logic,
            best_solver,
            best_par2_score,
            total_instances,
            solved_instances,
            solve_rate_percent,
            evaluation_name,
            evaluation_date
        FROM single_best_solver
        WHERE rank = 1
        """
        
        self.conn.execute(query_best)
        self.conn.commit()
        
        count_query = "SELECT COUNT(*) FROM single_best_final"
        count = self.conn.execute(count_query).fetchone()[0]
        print(f"   Created single best solver table with {count} logic types")

        
        if self.debug:
            self._debug_single_best_solver(logic_to_check='QF_ABVFPLRA')

        return count
    
    def create_virtual_best_solver_table(self):
        """
        Step 4: Calculate Virtual Best Solver performance
        For each benchmark instance, pick the best solver result
        """
        print("Step 4: Creating virtual best solver table...")
        
        # First, find the best solver for each benchmark instance
        query = """
        CREATE TEMP TABLE best_per_instance AS
        SELECT 
            logic,
            benchmark_name,
            query_id,
            MIN(par2_score) as best_par2_score,
            evaluation_name,
            evaluation_date
        FROM par2_results
        GROUP BY logic, benchmark_name, query_id, evaluation_name, evaluation_date
        """
        
        self.conn.execute(query)
        self.conn.commit()
        
        # Now calculate the virtual best solver performance per logic
        query_virtual = """
        CREATE TEMP TABLE virtual_best_solver AS
        SELECT 
            logic,
            'Virtual Best Solver' as solver_name,
            COUNT(*) as total_instances,
            AVG(best_par2_score) as avg_par2_score,
            evaluation_name,
            evaluation_date
        FROM best_per_instance
        GROUP BY logic, evaluation_name, evaluation_date
        """
        
        self.conn.execute(query_virtual)
        self.conn.commit()
        
        count_query = "SELECT COUNT(*) FROM virtual_best_solver"
        count = self.conn.execute(count_query).fetchone()[0]
        print(f"   Created virtual best solver table with {count} logic types")
        

        if self.debug:
            self._debug_virtual_best_solver(logic_to_check='QF_ABVFPLRA')

        return count
    
    def create_combined_ranking_table(self):
        """
        Step 5: Create final combined table with both single best and virtual best
        """
        print("Step 5: Creating combined ranking table...")
        
        query = """
        CREATE TEMP TABLE combined_solver_ranking AS
        SELECT 
            sbs.logic,
            sbs.best_solver as single_best_solver,
            sbs.best_par2_score as single_best_par2,
            sbs.total_instances as single_best_total_instances,
            sbs.solved_instances as single_best_solved_instances,
            sbs.solve_rate_percent as single_best_solve_rate,
            vbs.avg_par2_score as virtual_best_par2,
            vbs.total_instances as virtual_best_total_instances,
            ROUND(sbs.best_par2_score - vbs.avg_par2_score, 3) as performance_gap_difference,

            -- vvv ADD THESE LINES vvv
            lds.max_desc_length,
            lds.avg_desc_length,
            lds.stdev_desc_length,
            lds.min_desc_length,

            sbs.evaluation_name,
            sbs.evaluation_date
        FROM single_best_final sbs
        JOIN virtual_best_solver vbs ON sbs.logic = vbs.logic 
            AND sbs.evaluation_name = vbs.evaluation_name
        -- vvv ADD THIS JOIN vvv
        LEFT JOIN logic_description_stats lds ON sbs.logic = lds.logic
        ORDER BY sbs.logic
        """
        
        self.conn.execute(query)
        self.conn.commit()
        
        count_query = "SELECT COUNT(*) FROM combined_solver_ranking"
        count = self.conn.execute(count_query).fetchone()[0]
        print(f"   Created combined ranking table with {count} logic types")
        
        return count
    
    def get_single_best_solver_results(self):
        """Get results from single best solver analysis"""
        query = """
        SELECT 
            logic,
            best_solver as solver_name,
            best_par2_score as avg_par2_score,
            total_instances,
            solved_instances,
            solve_rate_percent,
            evaluation_name
        FROM single_best_final
        ORDER BY best_par2_score ASC
        """
                
        return pd.read_sql_query(query, self.conn)
    

    def _debug_single_best_solver(self, logic_to_check='QF_SLIA'):
        """Helper function to verify the single best solver selection."""
        print(f"\n--- DEBUG: Verifying Single Best Solver for Logic='{logic_to_check}' ---")
        
        # 1. Get the best solver identified by the script
        script_query = "SELECT * FROM single_best_final WHERE logic = ?;"
        script_df = pd.read_sql_query(script_query, self.conn, params=[logic_to_check])
        
        # 2. Manually find the best solver from the previous table
        manual_query = """
        SELECT solver_name, avg_par2_score FROM solver_logic_performance
        WHERE logic = ? ORDER BY avg_par2_score ASC LIMIT 1;
        """
        manual_df = pd.read_sql_query(manual_query, self.conn, params=[logic_to_check])
        
        print("Script's SBS Selection:")
        print(script_df.to_string(index=False))
        print("\nManual Verification (Top Solver):")
        print(manual_df.to_string(index=False))

        print("--- END: DEBUG ---\n")

    
    def get_virtual_best_solver_results(self):
        """Get results from virtual best solver analysis"""
        query = """
        SELECT 
            logic,
            solver_name,
            avg_par2_score,
            total_instances,
            evaluation_name
        FROM virtual_best_solver
        ORDER BY avg_par2_score ASC
        """
                
        return pd.read_sql_query(query, self.conn)
    
    
    def _debug_virtual_best_solver(self, logic_to_check='QF_SLIA'):
        """Helper function to verify the Virtual Best Solver calculation."""
        print(f"\n--- DEBUG: Verifying Virtual Best Solver for Logic='{logic_to_check}' ---")

        # 1. Get the VBS score from the script's table
        script_query = "SELECT avg_par2_score FROM virtual_best_solver WHERE logic = ?;"
        script_df = pd.read_sql_query(script_query, self.conn, params=[logic_to_check])

        # 2. Manually calculate the VBS score from the par2_results table
        manual_query = """
        SELECT AVG(min_score) as manual_vbs_par2
        FROM (
            SELECT MIN(par2_score) as min_score
            FROM par2_results
            WHERE logic = ?
            GROUP BY benchmark_name, query_id
        );
        """
        manual_df = pd.read_sql_query(manual_query, self.conn, params=[logic_to_check])

        print("Script's VBS Score:")
        print(script_df.to_string(index=False))
        print("\nManual Verification:")
        print(manual_df.to_string(index=False))
    
        print("--- END: DEBUG ---\n")
        
    def get_combined_ranking_results(self):
        """Get the final combined ranking results ordered by performance gap"""
        query = """
        SELECT 
            logic,
            single_best_solver,
            ROUND(single_best_par2, 3) as single_best_par2,
            single_best_total_instances,
            single_best_solved_instances,
            single_best_solve_rate,
            ROUND(virtual_best_par2, 3) as virtual_best_par2,
            virtual_best_total_instances,
            performance_gap_difference,

            -- vvv ADD THESE LINES vvv
            max_desc_length,
            ROUND(avg_desc_length, 2) AS avg_desc_length,
            ROUND(stdev_desc_length, 2) AS stdev_desc_length,
            min_desc_length,

            evaluation_name
        FROM combined_solver_ranking
        ORDER BY performance_gap_difference DESC
        """
        
        return pd.read_sql_query(query, self.conn)
    
    def get_detailed_solver_rankings_by_logic(self, logic_type):
        """Get detailed rankings for a specific logic type"""
        query = """
        SELECT 
            solver_name,
            ROUND(avg_par2_score, 3) as avg_par2_score,
            total_instances,
            solved_instances,
            ROUND(100.0 * solved_instances / total_instances, 2) as solve_rate_percent,
            evaluation_name
        FROM solver_logic_performance
        WHERE logic = ?
        ORDER BY avg_par2_score ASC
        """
        
        return pd.read_sql_query(query, self.conn, params=[logic_type])
    
    def run_full_analysis(self, year=None, incremental=None, save_results=True):
        """
        Run the complete analysis pipeline
        
        Args:
            year (int, optional): Year to filter analysis by
            incremental (bool, optional): 
                True = only incremental benchmarks
                False = only non-incremental benchmarks  
                None = include both types
            save_results (bool): Whether to save results to CSV files
        
        Returns:
            tuple: (single_best_df, virtual_best_df, combined_df)
        """
        benchmark_type = "incremental" if incremental is True else "non-incremental" if incremental is False else "all"
        print(f"Running PAR2-based solver ranking analysis for {year or 'all years'} ({benchmark_type} benchmarks)...")
        print("=" * 60)
        
        try:
            self.connect()
            
            # Run all steps
            self.create_par2_results_table(year, incremental)
            self.create_logic_description_stats_table()
            self.create_solver_logic_performance_table()
            self.create_single_best_solver_table()
            self.create_virtual_best_solver_table()
            self.create_combined_ranking_table()
            
            # Get results
            print("\nRetrieving results...")
            single_best_df = self.get_single_best_solver_results()
            virtual_best_df = self.get_virtual_best_solver_results()
            combined_df = self.get_combined_ranking_results()
            
            print(f"Single Best Solver results: {len(single_best_df)} logic types")
            print(f"Virtual Best Solver results: {len(virtual_best_df)} logic types")
            print(f"Combined ranking results: {len(combined_df)} logic types")
            
            # Save results if requested
            if save_results:
                
                # Ensure output directory exists
                import os
                os.makedirs("SBS_VBS", exist_ok=True)
                
                year_suffix = f"_{year}" if year else "_all_years"
                type_suffix = f"_{benchmark_type}" if incremental is not None else ""
                filename_suffix = f"{year_suffix}{type_suffix}"
                
                single_best_df.to_csv(f"SBS_VBS/single_best_solver{filename_suffix}.csv", index=False)
                virtual_best_df.to_csv(f"SBS_VBS/virtual_best_solver{filename_suffix}.csv", index=False)
                combined_df.to_csv(f"SBS_VBS/combined_solver_ranking{filename_suffix}.csv", index=False)
                
                print(f"\nResults saved to CSV files with suffix '{filename_suffix}'")
            
            return single_best_df, virtual_best_df, combined_df
            
        finally:
            self.disconnect()
    
    def analyze_specific_logic(self, logic_type, year=None, incremental=None):
        """
        Analyze solver rankings for a specific logic type
        
        Args:
            logic_type (str): Logic to analyze (e.g., 'QF_LIA')
            year (int, optional): Filter by year
            incremental (bool, optional): Filter by benchmark type
        """
        try:
            self.connect()
            self.create_par2_results_table(year, incremental)
            self.create_solver_logic_performance_table()
            
            results = self.get_detailed_solver_rankings_by_logic(logic_type)
            
            benchmark_type = "incremental" if incremental is True else "non-incremental" if incremental is False else "all"
            
            if not results.empty:
                print(f"\nSolver Rankings for {logic_type} ({benchmark_type} benchmarks):")
                print("=" * 60)
                print(results.to_string(index=False))
            else:
                print(f"No results found for logic type: {logic_type} with the specified filters")
            
            return results
            
        finally:
            self.disconnect()

def print_summary_report(single_best_df, virtual_best_df, combined_df):
    """Print a summary report of the analysis"""
    print("\n" + "=" * 80)
    print("SOLVER RANKING ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nTOP 5 LOGICS BY SINGLE BEST SOLVER PERFORMANCE (lowest PAR2):")
    print("-" * 60)
    top_single = combined_df.head(5)[['logic', 'single_best_solver', 'single_best_par2', 'single_best_solve_rate']]
    print(top_single.to_string(index=False))
    
    print(f"\nTOP 5 LOGICS BY VIRTUAL BEST SOLVER PERFORMANCE (lowest PAR2):")
    print("-" * 60)
    top_virtual = combined_df.head(5)[['logic', 'virtual_best_par2', 'virtual_best_total_instances']]
    print(top_virtual.to_string(index=False))
    
    print(f"\nLARGEST PERFORMANCE GAPS (Single Best vs Virtual Best):")
    print("-" * 60)
    largest_gaps = combined_df.nlargest(5, 'performance_gap_difference')[['logic', 'single_best_solver', 'performance_gap_difference']]
    print(largest_gaps.to_string(index=False))
    
    # Solver frequency analysis
    solver_counts = single_best_df['solver_name'].value_counts()
    print(f"\nMOST DOMINANT SOLVERS (appears as best in most logics):")
    print("-" * 60)
    print(solver_counts.head(10).to_string())

# Example usage
if __name__ == "__main__":
    # Initialize the ranking system
    db_path = "smtlib2025.sqlite"  # Update with your database path
    ranking = SolverRanking(db_path, debug=True)
        
    # Run analysis for 2024 non-incremental benchmarks
    print("Running analysis for 2024 incremental benchmarks...")
    single_best, virtual_best, combined = ranking.run_full_analysis(year=2024, incremental=False, save_results=True)
    
    # Print summary report
    print_summary_report(single_best, virtual_best, combined)
        
    
    """
    # Example: Analyze specific logic for non-incremental benchmarks
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS FOR QF_LIA LOGIC (NON-INCREMENTAL)")
    print("=" * 80)
    qf_lia_results = ranking.analyze_specific_logic("QF_LIA", 2024, incremental=False)
    """
    print("\nAnalysis complete! Check the generated CSV files for detailed results.")
