git clone https://clc-gitlab.cs.uiowa.edu:2443/SMT-LIB-benchmarks/QF_LRA.git
python ../scripts/dataset_create.py --split_size "250 500" --benchmark_dir  ./QF_LRA --dataset_dir ./qflra_exp
