#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || exit 2
run_dir="work_dirs/port_reliability_redc_size_v1/$run_id"
archive="work_dirs/port_reliability_redc_size_v1_${run_id}.tar.gz"
[[ ! -e "$run_dir" && ! -e "$archive" ]] || { printf 'Refuse overwrite\n' >&2; exit 2; }
mkdir -p "$run_dir"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  for device in 2 3; do
    used=$(nvidia-smi -i "$device" --query-gpu=memory.used --format=csv,noheader,nounits)
    utilization=$(nvidia-smi -i "$device" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if (( used < 200 && utilization < 5 )); then export CUDA_VISIBLE_DEVICES="$device"; break; fi
  done
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { printf 'GPU2/3 busy; stop without training\n' >&2; exit 3; }
fi
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
  tar -czf "$archive" -C work_dirs/port_reliability_redc_size_v1 "$run_id"
  sha256sum "$archive" | tee "$run_dir/archive.sha256"
  exit "$status"
}
trap finish EXIT
python -c 'import sys,torch,numpy; print(sys.version); print(torch.__version__,numpy.__version__); print(torch.cuda.get_device_name(0))' | tee "$run_dir/runtime.log"
python -m unittest discover -s tests -p test_port_reliability_redc_size_v1.py -v 2>&1 | tee "$run_dir/tests.log"
python crane_project/tools/run_port_reliability_redc_size_v1.py --gpu 0 --out "$run_dir/result" 2>&1 | tee "$run_dir/run.log"
