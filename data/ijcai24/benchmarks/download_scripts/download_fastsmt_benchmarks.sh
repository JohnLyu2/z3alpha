#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BENCHMARKS_DIR"

wget http://files.srl.inf.ethz.ch/data/smt_data.tar.gz

# Extract in script directory
tar -xzvf smt_data.tar.gz

mv smt_data/qf_bv/bruttomesso/core/train smt_data/qf_bv/bruttomesso/core/train1
mv smt_data/qf_bv/bruttomesso/core/valid smt_data/qf_bv/bruttomesso/core/train2
mv smt_data/qf_nia/AProVE/train smt_data/qf_nia/AProVE/train1
mv smt_data/qf_nia/AProVE/valid smt_data/qf_nia/AProVE/train2
mv smt_data/qf_nia/leipzig/train smt_data/qf_nia/leipzig/train1
mv smt_data/qf_nia/leipzig/valid smt_data/qf_nia/leipzig/train2
mv smt_data/qf_nra/hycomp/train smt_data/qf_nra/hycomp/train1
mv smt_data/qf_nra/hycomp/valid smt_data/qf_nra/hycomp/train2
mv smt_data/sage2/train smt_data/sage2/train1
mv smt_data/sage2/valid smt_data/sage2/train2

mv smt_data fastsmt_benchmarks
rm smt_data.tar.gz
