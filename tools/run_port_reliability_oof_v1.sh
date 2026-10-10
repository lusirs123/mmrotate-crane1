#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || exit 2
run_dir="work_dirs/port_reliability_oof_v1/$run_id"
archive="work_dirs/port_reliability_oof_v1_${run_id}.tar.gz"
[[ ! -e "$run_dir" && ! -e "$archive" ]] || { printf 'Refuse overwrite\n' >&2; exit 2; }
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
  tar --exclude='*.pth' --exclude='*/data' --exclude='*.pt' -czf "$archive" -C work_dirs/port_reliability_oof_v1 "$run_id"
  sha256sum "$archive" | tee "$run_dir/archive.sha256"
  exit "$status"
}
trap finish EXIT
gpu=""
for card in 3 2; do
  reading="$(nvidia-smi -i "$card" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  read -r used load <<< "${reading//,/ }"
  if (( used < 200 && load < 5 )); then gpu="$card"; break; fi
done
[[ -n "$gpu" ]] || { printf 'GPU2/3 busy; stop\n' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$gpu"
printf 'Physical GPU=%s\n' "$gpu" | tee "$run_dir/gpu.log"
python -c 'import sys,torch,numpy; print(sys.version,torch.__version__,numpy.__version__,torch.cuda.get_device_name(0))' | tee "$run_dir/runtime.log"
python -m unittest discover -s tests -p 'test_port_reliability_oof_v1.py' -v 2>&1 | tee "$run_dir/tests.log"
python crane_project/tools/run_port_reliability_oof_v1.py --mode all --gpu 0 --out "$run_dir/result" 2>&1 | tee "$run_dir/run.log"
python crane_project/tools/review_port_reliability_oof_v1.py "$run_dir/result" --receipt "$run_dir/result/server_review.json" 2>&1 | tee "$run_dir/review.log"
