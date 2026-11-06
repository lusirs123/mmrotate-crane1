"""Start only YOLO26 using the existing native training queue and guards.

The previously cancelled queue is left intact. This creates a fresh protocol;
it does not add a trainer, change the recipe, resume, or run GauCho again.
"""
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_external_obb_training_v1 as native

RESOURCE_POLICY = 'configs/yolo26/resource_policy_v1.json'


def yolo_only_protocol(protocol):
    if protocol['yolo26']['epochs'] != 24 or protocol['physical_gpus'] != [0,1]:
        raise ValueError('Keep the authorized 24-epoch GPU0/1 protocol')
    result = dict(protocol)
    result.update(native.read(ROOT/RESOURCE_POLICY))
    result.update(queue=['yolo26'],protocol='native_yolo26_only_formal_v1',
        request='New YOLO-only authorization; does not resume the cancelled 20261105 queue',
        result_requirements=dict(VAL_candidates=[16,18,20,22,24],
            native_checkpoint_names=['epoch15.pt','epoch17.pt','epoch19.pt','epoch21.pt','epoch23.pt'],
            selection='existing ckpt_sweep task rule; native best.pt is monitoring only',
            precision_TEST='one selected checkpoint, 1440 frames',
            efficiency='selected checkpoint; 50 warmup, 1440 frames x 3 repeats; mean/P95/FPS/parameters/memory',
            joint_accuracy_efficiency_report_required=True,
            automatic_offline_evaluation=False))
    return result


def launch(out):
    out = Path(out).resolve()
    # Reuse the already verified preparation, dataset/initialization checks,
    # native worker and memory guard; fix the queue before creating a process.
    native.prepare(out)
    protocol = yolo_only_protocol(native.read(out/'protocol.json'))
    name = str(Path(__file__).resolve().relative_to(ROOT))
    protocol['sources'] = dict(protocol['sources'], **{name:native.sha(ROOT/name)})
    protocol['sources'][RESOURCE_POLICY] = native.sha(ROOT/RESOURCE_POLICY)
    native.atomic(out/'protocol.json', protocol)
    command=[sys.executable,str(ROOT/'crane_project/tools/run_external_obb_training_v1.py'),
             '--stage','queue','--out',str(out)]
    with (out/'supervisor.log').open('x') as log:
        child=subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(supervisor_pid=child.pid,queue=['yolo26'],physical_gpus=[0,1],
        out=str(out),epochs=24,auto_TEST=False,
        joint_accuracy_efficiency_report_required=True,
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat())
    native.write_new(out/'launch.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();os.chdir(ROOT)
    import json
    print(json.dumps(launch(a.out),indent=2))


if __name__=='__main__':main()
