#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || { printf 'Invalid run id\n' >&2; exit 2; }
run_dir="work_dirs/port_reliability_opposite_border_v1/$run_id"
archive="work_dirs/port_reliability_opposite_border_v1_${run_id}.tar.gz"
[[ ! -e "$run_dir" && ! -e "$archive" ]] || { printf 'Refuse overwrite\n' >&2; exit 2; }
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
git diff --binary | sha256sum > "$run_dir/worktree_before.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_before.sha256"
python -m unittest discover -s tests -p test_port_reliability_opposite_border_v1.py -v 2>&1 | tee "$run_dir/tests.log"
python crane_project/tools/run_port_reliability_opposite_border_v1.py --stage train --out "$run_dir/train_check" 2>&1 | tee "$run_dir/train_check.log"
if python - "$run_dir/train_check/report.json" <<'PY'
import json,sys
sys.exit(0 if json.load(open(sys.argv[1]))['probe_gate']['passed'] else 1)
PY
then
  python crane_project/tools/run_port_reliability_opposite_border_v1.py --stage val --train-result "$run_dir/train_check" --out "$run_dir/val_check" 2>&1 | tee "$run_dir/val_check.log"
else
  printf 'Probe failed; VAL and TEST not evaluated. No retry or threshold search.\n' > "$run_dir/VAL_SKIPPED.txt"
fi
git diff --binary | sha256sum > "$run_dir/worktree_after.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_after.sha256"
cmp "$run_dir/worktree_before.sha256" "$run_dir/worktree_after.sha256"
cmp "$run_dir/index_before.sha256" "$run_dir/index_after.sha256"
tar -czf "$archive" -C work_dirs/port_reliability_opposite_border_v1 "$run_id"
sha256sum "$archive" | tee "$run_dir/archive.sha256"
