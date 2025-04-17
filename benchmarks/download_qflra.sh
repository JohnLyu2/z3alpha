#!/bin/bash

# Download QF_LRA benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_LRA.tar.zst?download=1" -O QF_LRA.tar.zst

# Extract the archive
zstd -d QF_LRA.tar.zst
tar xf QF_LRA.tar

# Remove the compressed files
rm QF_LRA.tar.zst QF_LRA.tar

python ../scripts/dataset_create.py --split_size "250 500" \
    --benchmark_dir  ./QF_LRA --dataset_dir ./qflra_exp
