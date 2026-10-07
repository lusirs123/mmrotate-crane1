#!/usr/bin/env bash
# Commands tracked by Git. Evidence grouped; archive stays at work_dirs root.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-all}"
run_id="${2:-$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$run_id" =~ ^[0-9]{8}_[0-9]{6}_[0-9]+$ ]] || { echo 'Invalid fresh run ID' >&2; exit 2; }
folder="work_dirs/port_geometry_size_boundary_v1/size_candidate_downstream_${run_id}"
if [[ "$stage" == pack ]]; then
  test -f "$folder/candidate/completion.json"
  test ! -f "$folder/candidate/failure.json"
  archive="work_dirs/port_size_candidate_downstream_v1_${run_id}.tar.gz"
  test ! -e "$archive"
  python - "$folder" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and f.suffix!='.pt' and f.name!='artifacts.json'}
with (p/'artifacts.json').open('x') as stream:json.dump(dict(files=files,excluded='GT-free roi.pt retained on server, bound by collect completion SHA; no weights in archive'),stream,indent=2)
PY
  tar --exclude='roi.pt' -czf "$archive" "$folder"
  sha256sum "$archive"
  exit 0
fi
[[ "$stage" == all || "$stage" == check ]] || { echo 'Use all, check, pack' >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES='' python tools/eval_port_size_candidate_downstream_v1.py --stage check --out-dir "$folder/check"
[[ "$stage" == check ]] && exit 0
gpu="$(python - <<'PY'
import subprocess
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
cards={int(i):(int(m),int(f),int(u)) for i,m,f,u in (s.split(',') for s in lines.splitlines())}
choice=next((i for i in (2,3,1,0) if i in cards and cards[i][0]<100 and cards[i][2]<10),None)
if choice is None:
    choice=next((i for i in (2,3) if i in cards and cards[i][1]>=4096),None)
if choice is None:raise SystemExit('No GPU with enough available memory; no task preemption')
print(choice)
PY
)"
echo "Actual candidate diagnosis uses physical GPU $gpu"
CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  nice -n 10 python tools/eval_port_size_candidate_downstream_v1.py --stage collect --gpu 0 --out-dir "$folder/collect"
# A separate process restores the candidate's original cuDNN8500 environment.
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 python tools/eval_port_size_candidate_downstream_v1.py \
  --stage candidate --gpu 0 --collect-dir "$folder/collect" --out-dir "$folder/candidate"
bash tools/run_port_size_candidate_downstream_v1.sh pack "$run_id"
