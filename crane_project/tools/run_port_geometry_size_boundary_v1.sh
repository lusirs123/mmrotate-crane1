#!/usr/bin/env bash
# Local source sync via Git; server-only finite run. Tarball at work_dirs root.
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# % 2 != 0 )); then
  echo 'Arguments are optional PATH/device pairs only' >&2
  exit 2
fi
BOUNDARY_INPUTS=("$@")
while (( $# )); do
  case "$1" in
    --collection-dir|--collection-archive|--cache-dir|--head-dir|--device) shift 2 ;;
    *) echo "Unsupported argument: $1" >&2; exit 2 ;;
  esac
done
BOUNDARY_NAME="port_geometry_size_boundary_v1_$(date +%Y%m%d_%H%M%S)_$$"
BOUNDARY_DIR="$PWD/work_dirs/port_geometry_size_boundary_v1/$BOUNDARY_NAME"
BOUNDARY_PACKAGE="$PWD/work_dirs/$BOUNDARY_NAME.tar.gz"
test ! -e "$BOUNDARY_DIR"
test ! -e "$BOUNDARY_PACKAGE"
mkdir -p "$BOUNDARY_DIR"
finish_boundary() {
  BOUNDARY_EXIT_CODE=$?
  trap - EXIT
  printf '%s\n' "$BOUNDARY_EXIT_CODE" > "$BOUNDARY_DIR/run_exit_code.txt"
  tar --exclude='*.pth' --exclude='*.pt' --exclude='*.pkl' \
    -czf "$BOUNDARY_PACKAGE" -C "$PWD/work_dirs/port_geometry_size_boundary_v1" "$BOUNDARY_NAME"
  sha256sum "$BOUNDARY_PACKAGE"
  exit "$BOUNDARY_EXIT_CODE"
}
trap finish_boundary EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
{
  python -m unittest discover -s tests -p test_port_geometry_size_boundary_v1.py -v
  python crane_project/tools/run_port_geometry_size_boundary_v1.py \
    --stage check --out-dir "$BOUNDARY_DIR/check"
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA required for this finite runner"'
  python crane_project/tools/run_port_geometry_size_boundary_v1.py \
    --stage prepare --out-dir "$BOUNDARY_DIR/prepare" "${BOUNDARY_INPUTS[@]}"
  python crane_project/tools/run_port_geometry_size_boundary_v1.py \
    --stage smoke --out-dir "$BOUNDARY_DIR/smoke" \
    --plan-file "$BOUNDARY_DIR/prepare/plan.json" "${BOUNDARY_INPUTS[@]}"
  python crane_project/tools/run_port_geometry_size_boundary_v1.py \
    --stage finite --out-dir "$BOUNDARY_DIR/finite" \
    --plan-file "$BOUNDARY_DIR/prepare/plan.json" \
    --smoke-dir "$BOUNDARY_DIR/smoke" "${BOUNDARY_INPUTS[@]}"
} 2>&1 | tee "$BOUNDARY_DIR/run.log"
