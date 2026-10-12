#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_id="${1:?run ID required}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || exit 2
run_dir="work_dirs/port_reliability_oof_v1/${run_id}_prevention_v1"
archive="work_dirs/port_reliability_oof_prevention_v1_${run_id}.tar.gz"
[[ ! -e "$run_dir" && ! -e "$archive" ]] || exit 2
mkdir -p "$run_dir"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
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
  tar --exclude='*.pth' --exclude='*.pt' --exclude='*/data' -czf "$archive" -C work_dirs/port_reliability_oof_v1 "${run_id}_prevention_v1"
  sha256sum "$archive" > "$run_dir/archive.sha256"
  exit "$status"
}
trap finish EXIT
idle_cards() {
  local apps card reading uuid used load
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
  for card in 2 3; do
    reading="$(nvidia-smi -i "$card" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
    read -r uuid used load <<< "${reading//,/ }"
    printf 'GPU%s %s used=%sMiB util=%s%%\n' "$card" "$uuid" "$used" "$load" >> "$run_dir/gpu.log"
    if [[ "$apps" == *"$uuid"* ]] || (( used >= 100 || load != 0 )); then
      printf 'GPU%s occupied; stop without training\n' "$card" >&2; return 3
    fi
  done
}
idle_cards
python -m unittest discover -s tests -p 'test_port_reliability_oof_prevention_v1.py' -v 2>&1 | tee "$run_dir/tests.log"
python -m unittest discover -s tests -p 'test_port_reliability_oof_gradient_chain_v1.py' -v 2>&1 | tee "$run_dir/budget_tests.log"
python crane_project/tools/run_port_reliability_oof_prevention_v1.py --mode prepare --out "$run_dir/result" 2>&1 | tee "$run_dir/prepare.log"
python crane_project/tools/run_port_reliability_oof_prevention_v1.py --mode check --out "$run_dir/result" 2>&1 | tee "$run_dir/check.log"
idle_cards
export CUDA_VISIBLE_DEVICES=3
python crane_project/tools/run_port_reliability_oof_prevention_v1.py --mode train --out "$run_dir/result" 2>&1 | tee "$run_dir/run.log"
