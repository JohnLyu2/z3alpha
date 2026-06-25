#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BENCHMARKS_DIR"

wget http://files.srl.inf.ethz.ch/data/smt_data.tar.gz

# Extract in script directory
tar -xzvf smt_data.tar.gz

merge_train_valid() {
    local dir="$1"
    mv "$dir/valid/"* "$dir/train/"
    rmdir "$dir/valid"
}

merge_train_valid smt_data/qf_bv/bruttomesso/core
merge_train_valid smt_data/qf_nia/AProVE
merge_train_valid smt_data/qf_nia/leipzig
merge_train_valid smt_data/qf_nra/hycomp
merge_train_valid smt_data/sage2

mv smt_data fastsmt_benchmarks
rm smt_data.tar.gz
