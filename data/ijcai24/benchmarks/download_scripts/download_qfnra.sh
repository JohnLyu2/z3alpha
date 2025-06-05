#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BENCHMARKS_DIR"

# Download QF_NRA benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_NRA.tar.zst?download=1" -O QF_NRA.tar.zst

# Extract the archive
zstd -d QF_NRA.tar.zst
tar xf QF_NRA.tar

# Remove the compressed files
rm QF_NRA.tar.zst QF_NRA.tar

# Move QF_NRA from non-incremental to benchmarks directory and clean up
mv non-incremental/QF_NRA .
rm -rf non-incremental

# create a all/ folder under QF_NRA
mkdir -p QF_NRA/all

# move all files under QF_NRA/ to all/ folder, excluding the all directory itself
for f in QF_NRA/*; do
    if [ "$f" != "QF_NRA/all" ]; then
        mv "$f" QF_NRA/all/
    fi
done

python "$BENCHMARKS_DIR/../../../scripts/dataset_create.py" --split_size "250 500" \
    --benchmark_dir  ./QF_NRA/all/ --dataset_dir ./QF_NRA/
