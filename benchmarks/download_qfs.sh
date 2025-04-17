#!/bin/bash

# Download QF_S benchmarks from Zenodo
wget "https://zenodo.org/records/10607722/files/QF_S.tar.zst?download=1" -O QF_S.tar.zst

# Extract the archive
zstd -d QF_S.tar.zst
tar xf QF_S.tar

# Remove the compressed files
rm QF_S.tar.zst QF_S.tar

python ../scripts/dataset_create.py --split_size "250 500" \
    --benchmark_dir  ./QF_S --dataset_dir ./qfs_exp
