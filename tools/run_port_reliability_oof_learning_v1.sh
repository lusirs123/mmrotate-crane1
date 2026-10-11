#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:?Explicit diagnostic RUN_ID required}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || exit 2
run_dir="work_dirs/port_reliability_oof_v1/$run_id"
[[ ! -e "$run_dir" ]] || { printf 'Refuse overwrite\n' >&2; exit 2; }
mkdir -p "$run_dir"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3
git diff --binary | sha256sum > "$run_dir/worktree_before.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_before.sha256"
finish() {
  status=$?
  trap - EXIT
  git diff --binary | sha256sum > "$run_dir/worktree_after.sha256"
  git diff --cached --binary | sha256sum > "$run_dir/index_after.sha256"
  cmp "$run_dir/worktree_before.sha256" "$run_dir/worktree_after.sha256" || status=4
  cmp "$run_dir/index_before.sha256" "$run_dir/index_after.sha256" || status=4
  printf '%s\n' "$status" > "$run_dir/exit_code.txt"
  exit "$status"
}
trap finish EXIT
check_idle() {
  local apps card uuid used load line
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
  for card in 2 3; do
    line="$(nvidia-smi -i "$card" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
    read -r uuid used load <<< "${line//,/ }"
    printf 'GPU%s used=%sMiB util=%s%%\n' "$card" "$used" "$load" >> "$run_dir/gpu.log"
    if [[ "$apps" == *"$uuid"* ]] || (( used >= 100 || load != 0 )); then
      printf 'GPU%s occupied; do not run diagnostic\n' "$card" >&2; return 3
    fi
  done
}
check_idle
python -m unittest discover -s tests -p 'test_port_reliability_oof_v1.py' -v > "$run_dir/oof_tests.log" 2>&1
python -m unittest discover -s tests -p 'test_port_reliability_oof_training_v2.py' -v > "$run_dir/repair_tests.log" 2>&1
python -m unittest discover -s tests -p 'test_port_reliability_oof_learning_v1.py' -v > "$run_dir/learning_tests.log" 2>&1
sleep 3
check_idle
python crane_project/tools/diagnose_port_reliability_oof_learning_v1.py --gpu 0 --source work_dirs/port_reliability_oof_v1/20261011_real_oof_v2_gpu3_fix1/result --out "$run_dir/result" > "$run_dir/run.log" 2>&1
