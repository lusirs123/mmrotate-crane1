#!/usr/bin/env bash
# Local source sync via Git; server-only finite run. Tarball at work_dirs root.
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# % 2 != 0 )); then
  echo 'Arguments are optional PATH/device pairs only' >&2
  exit 2
fi
CONTINUOUS_INPUTS=("$@")
while (( $# )); do
  case "$1" in
    --collection-dir|--collection-archive|--cache-dir|--head-dir|--device) shift 2 ;;
    *) echo "Unsupported argument: $1" >&2; exit 2 ;;
  esac
done
CONTINUOUS_NAME="port_geometry_size_boundary_continuous_v1_$(date +%Y%m%d_%H%M%S)_$$"
CONTINUOUS_DIR="$PWD/work_dirs/port_geometry_size_boundary_v1/$CONTINUOUS_NAME"
CONTINUOUS_PACKAGE="$PWD/work_dirs/$CONTINUOUS_NAME.tar.gz"
test ! -e "$CONTINUOUS_DIR"
test ! -e "$CONTINUOUS_PACKAGE"
mkdir -p "$CONTINUOUS_DIR"
finish_boundary() {
  CONTINUOUS_EXIT_CODE=$?
  trap - EXIT
  printf '%s\n' "$CONTINUOUS_EXIT_CODE" > "$CONTINUOUS_DIR/run_exit_code.txt"
  tar --exclude='*.pth' --exclude='*.pt' --exclude='*.pkl' \
    -czf "$CONTINUOUS_PACKAGE" -C "$PWD/work_dirs/port_geometry_size_boundary_v1" "$CONTINUOUS_NAME"
  sha256sum "$CONTINUOUS_PACKAGE"
  exit "$CONTINUOUS_EXIT_CODE"
}
trap finish_boundary EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
{
  python -m unittest discover -s tests -p test_port_geometry_size_boundary_continuous_v1.py -v
  python crane_project/tools/run_port_geometry_size_boundary_continuous_v1.py \
    --stage check --out-dir "$CONTINUOUS_DIR/check"
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA required for this finite runner"'
  python crane_project/tools/run_port_geometry_size_boundary_continuous_v1.py \
    --stage prepare --out-dir "$CONTINUOUS_DIR/prepare" "${CONTINUOUS_INPUTS[@]}"
  python crane_project/tools/run_port_geometry_size_boundary_continuous_v1.py \
    --stage smoke --out-dir "$CONTINUOUS_DIR/smoke" \
    --plan-file "$CONTINUOUS_DIR/prepare/plan.json" "${CONTINUOUS_INPUTS[@]}"
  python crane_project/tools/run_port_geometry_size_boundary_continuous_v1.py \
    --stage finite --out-dir "$CONTINUOUS_DIR/finite" \
    --plan-file "$CONTINUOUS_DIR/prepare/plan.json" \
    --smoke-dir "$CONTINUOUS_DIR/smoke" "${CONTINUOUS_INPUTS[@]}"
} 2>&1 | tee "$CONTINUOUS_DIR/run.log"
