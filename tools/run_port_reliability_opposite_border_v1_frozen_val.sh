#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
parent="work_dirs/port_reliability_opposite_border_v1/20261008_border_evidence_fix1"
out="$parent/val_supplement"
log="$parent/frozen_val_supplement.log"
tests_log="$parent/frozen_val_supplement_tests.log"
archive="work_dirs/port_reliability_opposite_border_v1_frozen_val_20261008.tar.gz"
[[ -d "$parent/train_check" && ! -e "$out" && ! -e "$archive" && ! -e "$log" && ! -e "$tests_log" ]] || { printf 'Refuse missing source or overwrite\n' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
tree_before=$(git diff --binary | sha256sum)
index_before=$(git diff --cached --binary | sha256sum)
python -m unittest discover -s tests -p test_port_reliability_opposite_border_v1_frozen_val.py -v 2>&1 | tee "$tests_log"
python crane_project/tools/run_port_reliability_opposite_border_v1_frozen_val.py --out "$out" 2>&1 | tee "$log"
mv "$log" "$out/run.log"
mv "$tests_log" "$out/tests.log"
printf '%s\n' "$tree_before" > "$out/worktree_before.sha256"
printf '%s\n' "$index_before" > "$out/index_before.sha256"
git diff --binary | sha256sum > "$out/worktree_after.sha256"
git diff --cached --binary | sha256sum > "$out/index_after.sha256"
cmp "$out/worktree_before.sha256" "$out/worktree_after.sha256"
cmp "$out/index_before.sha256" "$out/index_after.sha256"
tar -czf "$archive" -C "$parent" val_supplement
sha256sum "$archive" | tee "$out/archive.sha256"
