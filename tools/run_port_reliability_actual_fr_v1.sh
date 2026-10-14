#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_id="${1:?run id required}"
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
base="work_dirs/port_reliability_actual_fr_v1/$run_id"
[[ ! -e "$base" ]] || { echo 'Refuse overwrite'; exit 2; }
mkdir -p "$base"
python -m unittest discover -s tests -p test_port_reliability_actual_fr_v1.py -v 2>&1 | tee "$base/tests.log"
python crane_project/tools/run_port_reliability_actual_fr_v1.py --out "$base/result" 2>&1 | tee "$base/run.log"
python crane_project/tools/review_port_reliability_actual_fr_v1.py --out "$base/result" 2>&1 | tee "$base/review.log"
tar -czf "work_dirs/port_reliability_actual_fr_v1_${run_id}_analysis.tar.gz" "$base"
sha256sum "work_dirs/port_reliability_actual_fr_v1_${run_id}_analysis.tar.gz"
