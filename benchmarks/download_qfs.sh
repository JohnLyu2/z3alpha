#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Download QF_S benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_S.tar.zst?download=1" -O QF_S.tar.zst

# Extract the archive
zstd -d QF_S.tar.zst
tar xf QF_S.tar

# Remove the compressed files
rm QF_S.tar.zst QF_S.tar

# Move QF_S from non-incremental to current directory and clean up
mv non-incremental/QF_S .
rm -rf non-incremental

# create a all/ folder under QF_S
mkdir -p QF_S/all

# move all files under QF_S/ to all/ folder, excluding the all directory itself
for f in QF_S/*; do
    if [ "$f" != "QF_S/all" ]; then
        mv "$f" QF_S/all/
    fi
done

python ../scripts/dataset_create.py --split_size "250 500" \
    --benchmark_dir  ./QF_S/all/ --dataset_dir ./QF_S/
