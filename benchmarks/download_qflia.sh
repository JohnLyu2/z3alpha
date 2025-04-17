#!/bin/bash

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

python ../scripts/dataset_create.py --split_size "250 500" \
    --benchmark_dir  ./QF_LIA --dataset_dir ./qflia_exp
