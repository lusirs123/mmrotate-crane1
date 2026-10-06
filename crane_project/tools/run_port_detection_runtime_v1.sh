#!/usr/bin/env bash
# Run from an activated mmrotljj environment. All artifacts share one task root.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
runtime_gpu="${CUDA_VISIBLE_DEVICES:-3}"
task_name="four_arm_fix3_$(date +%Y%m%d_%H%M%S)_$$"
task_parent="work_dirs/port_detection_runtime_v1"
task_dir="$task_parent/$task_name"
mkdir -p "$task_parent"
mkdir "$task_dir"

pack_review() {
    task_status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$task_status" > "$task_dir/run_exit_code.txt"
    review_archive="work_dirs/${task_name}_review.tar.gz"
    tar -czf "$review_archive" -C "$task_parent" "$task_name"
    pack_status=$?
    if [ "$pack_status" -eq 0 ]; then
        sha256sum "$review_archive"
        printf 'Review archive: %s\n' "$project_root/$review_archive"
    else
        printf 'Packaging failed; retain task directory: %s\n' "$project_root/$task_dir" >&2
        if [ "$task_status" -eq 0 ]; then task_status=$pack_status; fi
    fi
    if [ "$task_status" -ne 0 ]; then
        printf 'Stopped with code %s; this is not a completed TEST speed result.\n' "$task_status" >&2
    fi
    exit "$task_status"
}
trap pack_review EXIT
printf 'Task directory: %s\nGPU visibility: %s\n' "$project_root/$task_dir" "$runtime_gpu" | tee "$task_dir/run.log"
python --version 2>&1 | tee -a "$task_dir/run.log"
python -c 'import torch, numpy, cv2' 2>&1 | tee -a "$task_dir/run.log"
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p test_port_detection_runtime_v1.py -v \
    2>&1 | tee "$task_dir/tests.log"
CUDA_VISIBLE_DEVICES="" python crane_project/tools/benchmark_port_detection_runtime_v1.py \
    --stage check --out-dir "$task_dir/check" 2>&1 | tee "$task_dir/check.log"
CUDA_VISIBLE_DEVICES="" python crane_project/tools/benchmark_port_detection_runtime_v1.py \
    --stage inputs --out-dir "$task_dir/inputs" 2>&1 | tee "$task_dir/inputs.log"
# The diagnosis-complete status alone is not an input-pass condition.
python - "$task_dir/inputs/input_diagnosis.json" <<'PY' 2>&1 | tee -a "$task_dir/run.log"
import json, sys
with open(sys.argv[1]) as stream:
    diagnosis = json.load(stream)['input_diagnosis']
for arm in ('eood', 'symeood'):
    candidates = [c for c in diagnosis[arm]['candidates'] if c.get('selection_exists')]
    if not candidates or not all(c.get('reviewed_input_contract', {}).get('passed') for c in candidates):
        raise SystemExit('Input identity failed for '+arm+'; see inputs.log')
for name, value in diagnosis['fixed_weights'].items():
    if not value['exists'] or value['actual_sha256'] != value['expected_sha256']:
        raise SystemExit('Fixed weight identity failed for '+name)
print('Reviewed detector input identities passed; proceeding to frozen TEST benchmark.')
PY
CUDA_VISIBLE_DEVICES="$runtime_gpu" python crane_project/tools/benchmark_port_detection_runtime_v1.py \
    --stage benchmark --gpu 0 --out-dir "$task_dir/benchmark" 2>&1 | tee "$task_dir/benchmark.log"
