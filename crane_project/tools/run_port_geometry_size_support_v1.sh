#!/usr/bin/env bash
# CPU numeric/annotation diagnosis only; archive always at work_dirs root.
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# != 0 && $# != 2 )); then
  echo 'Use no arguments, or --collection-dir PATH, or --collection-archive PATH' >&2
  exit 2
fi
if (( $# == 2 )) && [[ "$1" != '--collection-dir' && "$1" != '--collection-archive' ]]; then
  echo 'Only an existing collection input may be overridden' >&2
  exit 2
fi
SUPPORT_NAME="port_geometry_size_support_v1_$(date +%Y%m%d_%H%M%S)_$$"
SUPPORT_DIR="$PWD/work_dirs/$SUPPORT_NAME"
SUPPORT_PACKAGE="$PWD/work_dirs/$SUPPORT_NAME.tar.gz"
test ! -e "$SUPPORT_DIR"
test ! -e "$SUPPORT_PACKAGE"
mkdir -p "$SUPPORT_DIR"
finish_support() {
  SUPPORT_EXIT_CODE=$?
  trap - EXIT
  printf '%s\n' "$SUPPORT_EXIT_CODE" > "$SUPPORT_DIR/run_exit_code.txt"
  tar -czf "$SUPPORT_PACKAGE" -C "$PWD/work_dirs" "$SUPPORT_NAME"
  sha256sum "$SUPPORT_PACKAGE"
  exit "$SUPPORT_EXIT_CODE"
}
trap finish_support EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
{
  python -m unittest discover -s tests -p test_port_geometry_size_support_v1.py -v
  python crane_project/tools/diagnose_port_geometry_size_support_v1.py \
    --stage check --out-dir "$SUPPORT_DIR/check"
  python crane_project/tools/diagnose_port_geometry_size_support_v1.py \
    --stage diagnose --out-dir "$SUPPORT_DIR/diagnose" "$@"
} 2>&1 | tee "$SUPPORT_DIR/run.log"
