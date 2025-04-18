#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Download QF_LIA benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_LIA.tar.zst?download=1" -O QF_LIA.tar.zst

# Extract the archive
zstd -d QF_LIA.tar.zst
tar xf QF_LIA.tar

# Remove the compressed files
rm QF_LIA.tar.zst QF_LIA.tar

# Move QF_LIA from non-incremental to current directory and clean up
mv non-incremental/QF_LIA .
rm -rf non-incremental

# create a all/ folder under QF_LIA
mkdir -p QF_LIA/all

# move all files under QF_LIA/ to all/ folder, excluding the all directory itself
for f in QF_LIA/*; do
    if [ "$f" != "QF_LIA/all" ]; then
        mv "$f" QF_LIA/all/
    fi
done

python ../scripts/dataset_create.py --split_size "250 500" \
    --benchmark_dir  ./QF_LIA/all/ --dataset_dir ./QF_LIA/
