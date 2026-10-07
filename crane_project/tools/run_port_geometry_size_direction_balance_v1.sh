#!/usr/bin/env bash
# Fresh finite two-arm contrast. Sources/contracts stay in Git; tar at work_dirs root.
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# % 2 != 0 )); then echo 'Optional input PATH pairs only' >&2; exit 2; fi
DIRECTION_INPUTS=("$@")
while (( $# )); do
  case "$1" in
    --collection-dir|--collection-archive|--cache-dir|--head-dir) shift 2 ;;
    *) echo "Unsupported argument: $1" >&2; exit 2 ;;
  esac
done
DIRECTION_NAME="port_geometry_size_direction_balance_v1_$(date +%Y%m%d_%H%M%S)_$$"
DIRECTION_DIR="$PWD/work_dirs/port_geometry_size_boundary_v1/$DIRECTION_NAME"
DIRECTION_PACKAGE="$PWD/work_dirs/$DIRECTION_NAME.tar.gz"
test ! -e "$DIRECTION_DIR"; test ! -e "$DIRECTION_PACKAGE"
mkdir -p "$DIRECTION_DIR"
finish_direction() {
  DIRECTION_EXIT_CODE=$?
  trap - EXIT
  printf '%s\n' "$DIRECTION_EXIT_CODE" > "$DIRECTION_DIR/run_exit_code.txt"
  COPYFILE_DISABLE=1 tar --exclude='*.pth' --exclude='*.pt' --exclude='*.pkl' \
    -czf "$DIRECTION_PACKAGE" -C "$PWD/work_dirs/port_geometry_size_boundary_v1" "$DIRECTION_NAME"
  sha256sum "$DIRECTION_PACKAGE"
  exit "$DIRECTION_EXIT_CODE"
}
trap finish_direction EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
{
  env -u LD_LIBRARY_PATH python -m unittest discover -s tests -p test_port_geometry_size_direction_balance_v1.py -v
  env -u LD_LIBRARY_PATH python crane_project/tools/run_port_geometry_size_direction_balance_v1.py \
    --stage check --out-dir "$DIRECTION_DIR/check"
  env -u LD_LIBRARY_PATH python crane_project/tools/run_port_geometry_size_direction_balance_v1.py \
    --stage prepare --out-dir "$DIRECTION_DIR/prepare" "${DIRECTION_INPUTS[@]}"
  DIRECTION_GPU="$(python - <<'PY'
import os,subprocess
requested=os.environ.get('CUDA_VISIBLE_DEVICES')
if requested is not None and requested not in ('2','3'):
    raise SystemExit('Set CUDA_VISIBLE_DEVICES to physical GPU2 or3, or leave unset')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
cards={int(i):int(f) for i,f in (line.split(',') for line in rows.splitlines())}
order=[int(requested)] if requested else [2,3]
chosen=next((i for i in order if cards.get(i,0)>=4096),None)
if chosen is None:raise SystemExit('GPU2/3 have less than4096MiB free; no other task is interrupted')
print(chosen)
PY
)"
  echo "Direction contrast uses physical GPU $DIRECTION_GPU; original cuDNN8500 process"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$DIRECTION_GPU" nice -n 10 \
    python crane_project/tools/run_port_geometry_size_direction_balance_v1.py \
    --stage smoke --out-dir "$DIRECTION_DIR/smoke" --device cuda:0 \
    --plan-file "$DIRECTION_DIR/prepare/plan.json" "${DIRECTION_INPUTS[@]}"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$DIRECTION_GPU" nice -n 10 \
    python crane_project/tools/run_port_geometry_size_direction_balance_v1.py \
    --stage finite --out-dir "$DIRECTION_DIR/finite" --device cuda:0 \
    --plan-file "$DIRECTION_DIR/prepare/plan.json" --smoke-dir "$DIRECTION_DIR/smoke" "${DIRECTION_INPUTS[@]}"
} 2>&1 | tee "$DIRECTION_DIR/run.log"
