#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${CUDA_VISIBLE_DEVICES:?Select a free physical GPU2 or GPU3 before launch}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gpu}"
RUN_DIR="work_dirs/port_reliability_candidate_support_v1/$RUN_ID"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"
python -m unittest discover -s tests -p 'test_port_reliability_candidate_support_v1.py' -v 2>&1 | tee "$RUN_DIR/tests.log"
python crane_project/tools/run_port_reliability_candidate_support_v1.py --gpu 0 --out "$RUN_DIR/result" 2>&1 | tee "$RUN_DIR/run.log"
python crane_project/tools/review_port_reliability_candidate_support_v1.py --result "$RUN_DIR/result" --out "$RUN_DIR/result/server_review.json" 2>&1 | tee "$RUN_DIR/review.log"
