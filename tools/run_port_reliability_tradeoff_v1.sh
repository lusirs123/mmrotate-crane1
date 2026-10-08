#!/usr/bin/env bash
# CPU numeric reuse only. Keep all runs together; archives at work_dirs root.
set -euo pipefail
cd "$(dirname "$0")/.."
input_dir="${1:-work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1}"
run_id="${2:-$(date +%Y%m%d_%H%M%S)}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || { printf 'Invalid run id\n' >&2; exit 2; }
run_dir="work_dirs/port_reliability_tradeoff_v1/$run_id"
archive="work_dirs/port_reliability_tradeoff_v1_${run_id}.tar.gz"
[[ ! -e "$run_dir" && ! -e "$archive" ]] || { printf 'Refuse overwrite\n' >&2; exit 2; }
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES=""
git rev-parse HEAD > "$run_dir/git_commit.txt"
git diff --binary | sha256sum > "$run_dir/worktree_before.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_before.sha256"
python -m unittest discover -s tests -p test_port_reliability_tradeoff_v1.py -v 2>&1 | tee "$run_dir/tests.log"
python crane_project/tools/analyze_port_reliability_tradeoff_v1.py \
  --input-dir "$input_dir" --out "$run_dir/analysis" 2>&1 | tee "$run_dir/run.log"
git diff --binary | sha256sum > "$run_dir/worktree_after.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_after.sha256"
cmp "$run_dir/worktree_before.sha256" "$run_dir/worktree_after.sha256"
cmp "$run_dir/index_before.sha256" "$run_dir/index_after.sha256"
tar -czf "$archive" -C work_dirs/port_reliability_tradeoff_v1 "$run_id"
sha256sum "$archive" | tee "$run_dir/archive.sha256"
