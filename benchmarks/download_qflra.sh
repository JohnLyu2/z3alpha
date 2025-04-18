#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Download QF_LRA benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_LRA.tar.zst?download=1" -O QF_LRA.tar.zst

# Extract the archive
zstd -d QF_LRA.tar.zst
tar xf QF_LRA.tar

# Remove the compressed files
rm QF_LRA.tar.zst QF_LRA.tar

# Move QF_LRA from non-incremental to current directory and clean up
mv non-incremental/QF_LRA .
rm -rf non-incremental

# create a all/ folder under QF_LRA
mkdir -p QF_LRA/all

# move all files under QF_LRA/ to all/ folder, excluding the all directory itself
for f in QF_LRA/*; do
    if [ "$f" != "QF_LRA/all" ]; then
        mv "$f" QF_LRA/all/
    fi
done

python ../scripts/dataset_create.py --split_size "250 500" \
    --benchmark_dir  ./QF_LRA/all/ --dataset_dir ./QF_LRA/
