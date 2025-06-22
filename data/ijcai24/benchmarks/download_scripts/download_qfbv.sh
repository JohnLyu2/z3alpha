#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BENCHMARKS_DIR"

# Download QF_BV benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_BV.tar.zst?download=1" -O QF_BV.tar.zst

# Extract the archive
zstd -d QF_BV.tar.zst
tar xf QF_BV.tar

# Remove the compressed files
rm QF_BV.tar.zst QF_BV.tar

# Move QF_BV from non-incremental to benchmarks directory and clean up
mv non-incremental/QF_BV .
rm -rf non-incremental

# create a all/ folder under QF_BV
mkdir -p QF_BV/all

# move all files under QF_BV/ to all/ folder, excluding the all directory itself
for f in QF_BV/*; do
    if [ "$f" != "QF_BV/all" ]; then
        mv "$f" QF_BV/all/
    fi
done

python "$BENCHMARKS_DIR/../../../scripts/dataset_create.py" --split_size "250 500" \
    --benchmark_dir  ./QF_BV/all/ --dataset_dir ./QF_BV/
