#!/usr/bin/env bash
# New finite protocol only. Result archive at work_dirs root; weights stay server.
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# % 2 != 0 )); then
  echo 'Arguments are optional PATH/device pairs only' >&2
  exit 2
fi
CONTRAST_INPUTS=("$@")
while (( $# )); do
  case "$1" in
    --collection-dir|--collection-archive|--cache-dir|--head-dir|--device) shift 2 ;;
    *) echo "Unsupported argument: $1" >&2; exit 2 ;;
  esac
done
CONTRAST_NAME="port_geometry_size_contrast_v1_$(date +%Y%m%d_%H%M%S)_$$"
CONTRAST_DIR="$PWD/work_dirs/$CONTRAST_NAME"
CONTRAST_PACKAGE="$PWD/work_dirs/$CONTRAST_NAME.tar.gz"
test ! -e "$CONTRAST_DIR"
test ! -e "$CONTRAST_PACKAGE"
mkdir -p "$CONTRAST_DIR"
finish_contrast() {
  CONTRAST_EXIT_CODE=$?
  trap - EXIT
  printf '%s\n' "$CONTRAST_EXIT_CODE" > "$CONTRAST_DIR/run_exit_code.txt"
  tar --exclude='*.pth' --exclude='*.pt' --exclude='*.pkl' \
    -czf "$CONTRAST_PACKAGE" -C "$PWD/work_dirs" "$CONTRAST_NAME"
  sha256sum "$CONTRAST_PACKAGE"
  exit "$CONTRAST_EXIT_CODE"
}
trap finish_contrast EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
{
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA required for this finite runner"'
  python -m unittest discover -s tests -p test_port_geometry_size_contrast_v1.py -v
  python crane_project/tools/run_port_geometry_size_contrast_v1.py \
    --stage prepare --out-dir "$CONTRAST_DIR/prepare" "${CONTRAST_INPUTS[@]}"
  python crane_project/tools/run_port_geometry_size_contrast_v1.py \
    --stage smoke --out-dir "$CONTRAST_DIR/smoke" \
    --plan-file "$CONTRAST_DIR/prepare/plan.json" "${CONTRAST_INPUTS[@]}"
  python crane_project/tools/run_port_geometry_size_contrast_v1.py \
    --stage finite --out-dir "$CONTRAST_DIR/finite" \
    --plan-file "$CONTRAST_DIR/prepare/plan.json" \
    --smoke-dir "$CONTRAST_DIR/smoke" "${CONTRAST_INPUTS[@]}"
} 2>&1 | tee "$CONTRAST_DIR/run.log"
