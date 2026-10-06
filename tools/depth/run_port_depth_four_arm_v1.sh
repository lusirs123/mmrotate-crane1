#!/usr/bin/env bash
# One task directory and one root-level archive; fail without exiting parent SSH.
set -euo pipefail
cd "$(dirname "$0")/../.."
DEPTH_TASK_NAME="port_depth_four_arm_v1_train03_$(date +%Y%m%d_%H%M%S)_$$"
DEPTH_TASK_ROOT="$PWD/work_dirs/port_depth_four_arm_v1/$DEPTH_TASK_NAME"
DEPTH_PACKAGE="$PWD/work_dirs/${DEPTH_TASK_NAME}.tar.gz"
test ! -e "$DEPTH_TASK_ROOT"
test ! -e "$DEPTH_PACKAGE"
mkdir -p "$DEPTH_TASK_ROOT"
finish_depth() {
  DEPTH_EXIT_CODE=$?
  trap - EXIT
  printf '%s\n' "$DEPTH_EXIT_CODE" > "$DEPTH_TASK_ROOT/run_exit_code.txt"
  tar -czf "$DEPTH_PACKAGE" -C "$(dirname "$DEPTH_TASK_ROOT")" "$DEPTH_TASK_NAME"
  sha256sum "$DEPTH_PACKAGE"
  exit "$DEPTH_EXIT_CODE"
}
trap finish_depth EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
{
python -c 'import torch, numpy, cv2, mmcv, mmdet, mmrotate'
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p test_port_depth_four_arm_v1.py -v
CUDA_VISIBLE_DEVICES="" python tools/depth/audit_port_depth_four_arm_v1.py \
  --stage check --out-dir "$DEPTH_TASK_ROOT/check"
python tools/depth/audit_port_depth_four_arm_v1.py \
  --stage audit --gpu 0 --out-dir "$DEPTH_TASK_ROOT/audit" "$@"
} 2>&1 | tee "$DEPTH_TASK_ROOT/run.log"
