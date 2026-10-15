#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
version=port_simple_pilot_increment_v1
run_id="${1:?run id required}"
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
base="work_dirs/$version/$run_id"
[[ ! -e "$base" && ! -L "$base" ]] || exit 2
mkdir -p "$base"
git diff --binary | sha256sum > "$base/worktree_before.sha256"
git diff --cached --binary | sha256sum > "$base/index_before.sha256"
python -c 'import json,os,sys,subprocess;print(json.dumps(dict(cwd=os.path.realpath(os.getcwd()),environment=os.environ.get("CONDA_DEFAULT_ENV"),python=sys.executable,HEAD=subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip())))' > "$base/runtime.json"
python -m unittest discover -s tests -p "test_${version}.py" -v 2>&1 | tee "$base/tests.log"
python "crane_project/tools/run_${version}.py" --out "$base/result" 2>&1 | tee "$base/run.log"
python "crane_project/tools/review_port_simple_pilot_increment_v1.py" --out "$base/result" --receipt independent_review_server.json
git diff --binary | sha256sum > "$base/worktree_after.sha256"
git diff --cached --binary | sha256sum > "$base/index_after.sha256"
cmp "$base/worktree_before.sha256" "$base/worktree_after.sha256"
cmp "$base/index_before.sha256" "$base/index_after.sha256"
archive="work_dirs/${version}_${run_id}_analysis.tar.gz"
[[ ! -e "$archive" ]] || exit 2
tar -czf "$archive" "$base"
sha256sum "$archive"
