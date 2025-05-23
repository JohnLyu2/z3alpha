#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BENCHMARKS_DIR"

# first download QF_NRA benchmarks from Zenodo
# https://zenodo.org/records/10607722/files/QF_NRA.tar.zst?download=1
wget "https://zenodo.org/records/10607722/files/QF_NRA.tar.zst?download=1" -O QF_NRA.tar.zst

# extract the archive
zstd -d QF_NRA.tar.zst
tar xf QF_NRA.tar

# remove the compressed files
rm QF_NRA.tar.zst QF_NRA.tar

# move non-incremental/QF_NRA/LassoRanker to benchmarks directory and clean up
mv non-incremental/QF_NRA/LassoRanker .
rm -rf non-incremental

# create a all/ folder under LassoRanker
mkdir -p LassoRanker/all

# move all files under LassoRanker/ to all/ folder, excluding the all/ directory itself
for f in LassoRanker/*; do
    if [ "$f" != "LassoRanker/all" ]; then
        mv "$f" LassoRanker/all/
    fi
done

# run the dataset creation script
python "$BENCHMARKS_DIR/../../../scripts/dataset_create.py" --split_size "250 250" \
    --benchmark_dir ./LassoRanker/all/ --dataset_dir ./LassoRanker/
