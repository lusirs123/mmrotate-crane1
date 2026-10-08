#!/usr/bin/env bash
# Existing paired predictions, CPU only; all artifacts in one run folder.
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || { printf 'Invalid run id\n' >&2; exit 2; }
run_dir="work_dirs/port_reliability_scale_asymmetry_v1/$run_id"
archive="work_dirs/port_reliability_scale_asymmetry_v1_${run_id}.tar.gz"
[[ ! -e "$run_dir" && ! -e "$archive" ]] || { printf 'Refuse overwrite\n' >&2; exit 2; }
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
git diff --binary | sha256sum > "$run_dir/worktree_before.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_before.sha256"
python -m unittest discover -s tests -p test_port_reliability_scale_asymmetry_v1.py -v 2>&1 | tee "$run_dir/tests.log"
python crane_project/tools/run_port_reliability_scale_asymmetry_v1.py --out "$run_dir/train_check" 2>&1 | tee "$run_dir/train_check.log"
git diff --binary | sha256sum > "$run_dir/worktree_after.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_after.sha256"
cmp "$run_dir/worktree_before.sha256" "$run_dir/worktree_after.sha256"
cmp "$run_dir/index_before.sha256" "$run_dir/index_after.sha256"
tar -czf "$archive" -C work_dirs/port_reliability_scale_asymmetry_v1 "$run_id"
sha256sum "$archive" | tee "$run_dir/archive.sha256"
