#!/usr/bin/env bash

echo "Adding indexes"
echo "    on file" "$1"

echo "Add index for Benchmarks table"
sqlite3 "$1" "create index benchIdx1 on Benchmarks(name, family, logic, isIncremental);"

echo "Add index for Queries table"
sqlite3 "$1" "create index benchIdx2 on Queries(benchmark);"

echo "Add index for Families table"
sqlite3 "$1" "create index benchIdx3 on Families(name, folderName, firstOccurrence);"

echo "Add index for Symbols table"
sqlite3 "$1" "create index evalIdx1 on Symbols(name);"

echo "Add index for SymbolCounts table"
sqlite3 "$1" "create index evalIdx2 on SymbolCounts(symbol, query, count);"

echo "Add index for SolverVariants table"
sqlite3 "$1" "create index evalIdx4 on SolverVariants(solver);"

echo "Add index for Results table"
sqlite3 "$1" "create index evalIdx5 on Results(query, solverVariant, status, evaluation);"

echo "Add index for Evaluations table"
sqlite3 "$1" "create index evalIdx6 on Evaluations(date);"

echo "Add index for Ratings table"
sqlite3 "$1" "create index evalIdx7 on Ratings(query, evaluation);"

echo "Call SQLite analyze."
sqlite3 "$1" "analyze;"
