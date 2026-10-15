#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
run_id="${1:?run id required}"
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
version=port_reliability_response_warm_test_v1
base="work_dirs/$version/$run_id"
archive="work_dirs/${version}_${run_id}_analysis.tar.gz"
[[ "$PWD" == /media/omnisky/personal_files/ljj/symEOOD && "${CONDA_DEFAULT_ENV:-}" == mmrotljj && ! -e "$base" && ! -e "$archive" ]] || exit 2
gpu=3
reading="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
read -r used load <<< "${reading//,/ }"
(( used < 200 && load < 5 )) || exit 3
export CUDA_VISIBLE_DEVICES="$gpu"
mkdir -p "$base"
git diff --binary | sha256sum > "$base/worktree_before.sha256"
git diff --cached --binary | sha256sum > "$base/index_before.sha256"
finish() {
  code=$?
  trap - EXIT
  git diff --binary | sha256sum > "$base/worktree_after.sha256"
  git diff --cached --binary | sha256sum > "$base/index_after.sha256"
  cmp "$base/worktree_before.sha256" "$base/worktree_after.sha256" || code=4
  cmp "$base/index_before.sha256" "$base/index_after.sha256" || code=4
  printf '%s\n' "$code" > "$base/exit_code.txt"
  tar -czf "$archive" "$base"
  sha256sum "$archive" > "$base/archive.sha256"
  exit "$code"
}
trap finish EXIT
printf 'Physical GPU %s; no training; frozen seed1701\n' "$gpu" | tee "$base/gpu.log"
python -m unittest discover -s tests -p 'test_port_reliability_response_warm_test_v1.py' -v 2>&1 | tee "$base/tests.log"
python crane_project/tools/eval_port_reliability_response_warm_test_v1.py --out "$base/result" 2>&1 | tee "$base/run.log"
python crane_project/tools/eval_port_reliability_response_warm_test_v1.py --out "$base/result" --review --receipt independent_review_server.json 2>&1 | tee "$base/review.log"
