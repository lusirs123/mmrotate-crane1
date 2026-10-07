#!/usr/bin/env bash
# Source/commands stay in Git; generated evidence stays in existing grouped dirs.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-all}"
run_id="${2:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ ! "$run_id" =~ ^[0-9]{8}_[0-9]{6}_[0-9]+$ ]]; then
  echo 'Expected a fresh YYYYMMDD_HHMMSS_PID run ID' >&2; exit 2
fi
depth_dir="work_dirs/port_depth_four_arm_v1/frozen_downstream_${run_id}"
reliability_dir="work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/frozen_test_${run_id}"
check_dir="work_dirs/port_depth_four_arm_v1/frozen_downstream_check_${run_id}"
if [[ "$stage" == pack ]]; then
  test -f "$depth_dir/completion.json"
  test -f "$reliability_dir/completion.json"
  archive="work_dirs/port_frozen_downstream_v1_${run_id}.tar.gz"
  test ! -e "$archive"
  tar -czf "$archive" "$depth_dir" "$reliability_dir" "$check_dir"
  sha256sum "$archive"
  exit 0
fi
[[ "$stage" == all || "$stage" == check ]] || { echo 'Use all, check or pack' >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Reuse the original head's cuDNN 8801; Python validates all runtime versions.
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES='' python tools/eval_port_frozen_downstream_v1.py --stage check --out-dir "$check_dir"
[[ "$stage" == check ]] && exit 0
CUDA_VISIBLE_DEVICES='' python tools/eval_port_frozen_downstream_v1.py --stage reliability \
  --out-dir "$reliability_dir" \
  --geometry-test-dir work_dirs/port_results/geometry/port_geometry_midpoint_sigma15_v1_test_eval \
  --metadata-dir work_dirs/port_reliability_branches_v1_test_cached_v1
gpu="$(python - <<'PY'
import subprocess
values=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
idle={int(a):int(b)<100 and int(c)<10 for a,b,c in (line.split(',') for line in values.splitlines())}
for i in (2,3,1,0):
    if idle.get(i):print(i);break
else:raise SystemExit('No idle GPU; existing tasks will not be interrupted')
PY
)"
echo "Frozen depth evaluation uses physical GPU $gpu (prefer idle 2/3)."
CUDA_VISIBLE_DEVICES="$gpu" python tools/eval_port_frozen_downstream_v1.py \
  --stage depth --out-dir "$depth_dir" --data-root crane_project/data/webots_depth --gpu 0
echo "FROZEN_DOWNSTREAM_COMPLETE run_id=$run_id"
bash tools/run_port_frozen_downstream_v1.sh pack "$run_id"
