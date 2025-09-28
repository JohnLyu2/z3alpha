#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BENCHMARKS_DIR"

# Download QF_NIA benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_NIA.tar.zst?download=1" -O QF_NIA.tar.zst

# Extract the archive
zstd -d QF_NIA.tar.zst
tar xf QF_NIA.tar

# Remove the compressed files
rm QF_NIA.tar.zst QF_NIA.tar

# Move QF_NIA from non-incremental to benchmarks directory and clean up
mv non-incremental/QF_NIA .
rm -rf non-incremental

# create a all/ folder under QF_NIA
mkdir -p QF_NIA/all

# move all files under QF_NIA/ to all/ folder, excluding the all directory itself
for f in QF_NIA/*; do
    if [ "$f" != "QF_NIA/all" ]; then
        mv "$f" QF_NIA/all/
    fi
done

python "$BENCHMARKS_DIR/../../../scripts/dataset_create.py" --split_size "250 500" \
    --benchmark_dir  ./QF_NIA/all/ --dataset_dir ./QF_NIA/
