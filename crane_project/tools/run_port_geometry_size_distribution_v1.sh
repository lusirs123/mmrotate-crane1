#!/usr/bin/env bash
# Single-GPU cache-only run. Input root is the preserved existing server checkout.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
: "${SYME_INPUT_ROOT:?Set the existing SymEOOD server checkout}"
: "${SYME_RUN_ID:?Set a unique run id}"
: "${CUDA_VISIBLE_DEVICES:?Select exactly one GPU}"
base="work_dirs/port_geometry_size_distribution_v1"
launch="$base/${SYME_RUN_ID}_launch"
check="$base/${SYME_RUN_ID}_check"
run="$base/${SYME_RUN_ID}_run"
archive="work_dirs/${SYME_RUN_ID}_evidence.tar.gz"
if [[ -e "$launch" || -e "$check" || -e "$run" || -e "$archive" ]]; then
  echo 'Refuse existing run/transfer paths' >&2
  exit 2
fi
mkdir -p "$launch"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$launch/run_exit_code.txt"
  git rev-parse HEAD > "$launch/git_head.txt"
  git status --short > "$launch/git_status.txt"
  items=("$launch")
  [[ ! -d "$check" ]] || items+=("$check")
  [[ ! -d "$run" ]] || items+=("$run")
  tar -czf "$archive" "${items[@]}"
  sha256sum "$archive" > "$archive.sha256"
  printf 'EXIT=%s ARCHIVE=%s\n' "$code" "$archive"
  exit "$code"
}
trap finish EXIT
python -m pytest -q tests/test_port_geometry_size_distribution_v1.py 2>&1 | tee "$launch/tests.log"
python -u crane_project/tools/run_port_geometry_size_distribution_v1.py --stage check --out-dir "$check" 2>&1 | tee "$launch/check.log"
python -u crane_project/tools/run_port_geometry_size_distribution_v1.py --stage run --out-dir "$run" \
 --cache-dir "$SYME_INPUT_ROOT/work_dirs/port_geometry_midpoint_formal_v1_roi_cache" \
 --head-dir "$SYME_INPUT_ROOT/work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5" \
 --b-weight "$SYME_INPUT_ROOT/work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth" 2>&1 | tee "$launch/run.log"
