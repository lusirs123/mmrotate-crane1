#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN_ID="${1:-20261014_frozen_stack_test_v1}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
DEST="work_dirs/port_reliability_temporal_stack_test_v1/$RUN_ID"
[[ ! -e "$DEST" ]] || { echo 'Refuse overwrite'; exit 2; }
mkdir -p "$DEST"
python -m unittest discover -s tests -p 'test_port_reliability_temporal_stack_test_v1.py' -v 2>&1 | tee "$DEST/tests.log"
python crane_project/tools/eval_port_reliability_temporal_stack_test_v1.py --out "$DEST/result" 2>&1 | tee "$DEST/run.log"
python crane_project/tools/review_port_reliability_temporal_stack_test_v1.py --out "$DEST/result" 2>&1 | tee "$DEST/review.log"
tar -czf "work_dirs/port_reliability_temporal_stack_test_v1_${RUN_ID}_analysis.tar.gz" "$DEST"
