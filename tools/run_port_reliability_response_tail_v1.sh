#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
version=port_reliability_response_tail_v1
run_id="${1:?run id required}"
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
base="work_dirs/$version/$run_id"
archive="work_dirs/${version}_${run_id}_analysis.tar.gz"
[[ ! -e "$base" && ! -e "$archive" ]] || exit 2
[[ "$PWD" == /media/omnisky/personal_files/ljj/symEOOD && "${CONDA_DEFAULT_ENV:-}" == mmrotljj ]] || exit 2
gpu=""
for card in 3 2; do
  reading="$(nvidia-smi -i "$card" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  read -r used load <<< "${reading//,/ }"
  if (( used < 200 && load < 5 )); then gpu="$card"; break; fi
done
[[ -n "$gpu" ]] || { printf 'GPU2/3 busy; no run started\n' >&2; exit 3; }
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
python -m unittest discover -s tests -p 'test_port_reliability_response_tail_v1.py' -v 2>&1 | tee "$base/tests.log"
python crane_project/tools/run_port_reliability_response_tail_v1.py --out "$base/result" 2>&1 | tee "$base/run.log"
python crane_project/tools/review_port_reliability_response_tail_v1.py --out "$base/result" --receipt independent_review_server.json 2>&1 | tee "$base/review.log"
