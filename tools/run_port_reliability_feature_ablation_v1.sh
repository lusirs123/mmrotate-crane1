#!/usr/bin/env bash
# Fixed CPU fits only; gated cached TEST, one grouped parent, archives at root.
set -euo pipefail
cd "$(dirname "$0")/.."
input_dir="${1:-work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1}"
run_id="${2:-$(date +%Y%m%d_%H%M%S)}"
[[ "$run_id" =~ ^[A-Za-z0-9_]+$ ]] || { printf 'Invalid run id\n' >&2; exit 2; }
run_dir="work_dirs/port_reliability_feature_ablation_v1/$run_id"
archive="work_dirs/port_reliability_feature_ablation_v1_${run_id}.tar.gz"
[[ ! -e "$run_dir" && ! -e "$archive" ]] || { printf 'Refuse overwrite\n' >&2; exit 2; }
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES=""
git rev-parse HEAD > "$run_dir/git_commit.txt"
git diff --binary | sha256sum > "$run_dir/worktree_before.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_before.sha256"
python -m unittest discover -s tests -p test_port_reliability_feature_ablation_v1.py -v 2>&1 | tee "$run_dir/tests.log"
python crane_project/tools/run_port_reliability_feature_ablation_v1.py fit \
  --input-dir "$input_dir" --out "$run_dir/fit" 2>&1 | tee "$run_dir/fit.log"
selected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_arm"] or "")' "$run_dir/fit/completion.json")"
if [[ -n "$selected" ]]; then
  python crane_project/tools/run_port_reliability_feature_ablation_v1.py test \
    --fit-dir "$run_dir/fit" --out "$run_dir/test" 2>&1 | tee "$run_dir/test.log"
else
  printf 'VAL did not select a candidate; TEST not opened; original policy retained.\n' | tee "$run_dir/TEST_SKIPPED.txt"
fi
git diff --binary | sha256sum > "$run_dir/worktree_after.sha256"
git diff --cached --binary | sha256sum > "$run_dir/index_after.sha256"
cmp "$run_dir/worktree_before.sha256" "$run_dir/worktree_after.sha256"
cmp "$run_dir/index_before.sha256" "$run_dir/index_after.sha256"
tar -czf "$archive" -C work_dirs/port_reliability_feature_ablation_v1 "$run_id"
sha256sum "$archive" | tee "$run_dir/archive.sha256"
