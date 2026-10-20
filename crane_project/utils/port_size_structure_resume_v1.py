"""S046 I/O-only recovery. Immutable epochs stay exclusive; status is atomic."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

# Exact failed release. Only runner/status I/O and its tests were repaired.
FAILED_RUNNER_SHA='dcc2b8ec030c0d5fe3f74cd80c87e9c5ee7801050ffba2dfe5ff911a592d6645'
FAILED_TEST_SHA='53feb17ad50e817123a4916878fdfb2458917b5cd00d50b54d9e59d80443985e'
ALLOWED_OLD={'crane_project/tools/run_port_size_structure_v1.py':FAILED_RUNNER_SHA,
             'tests/test_port_size_structure_v1.py':FAILED_TEST_SHA}
ARMS=('V_structure','D_direct')
BASE=['engineering.json','ENGINEERING_PASSED.json','train_support_views.json','train_support_summary.json',
      'baseline_val.rows.jsonl.gz','baseline_train.rows.jsonl.gz','baseline_val.summary.json']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def status_write(path,value):
    """Only mutable progress/training status; never epoch/weight/source files."""
    path=Path(path)
    if path.name not in ('progress.json','training.json'):raise ValueError('Immutable artifact must not be replaced')
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def prepare(previous,out,protocol,proof):
    """Verify the failed run and copy only SHA-bound complete epoch prefixes.

    Copy leaves original failure artifacts byte-identical. Metadata/status from
    the failed writer are not used to decide completed epochs.
    """
    previous,out=Path(previous),Path(out)
    if previous==out or previous.parent!=out.parent:raise ValueError('Fresh sibling recovery directory required')
    index=read(previous/'artifacts.json')['files']
    if any(sha(previous/name)!=h for name,h in index.items()):raise ValueError('Failed artifacts changed')
    finished=read(previous/'finished.json')
    if finished['status']!='FAILED' or 'FileExistsError' not in finished.get('error',''):raise ValueError('Not the pinned I/O failure')
    if read(previous/'protocol.json')!=protocol:raise ValueError('Training protocol differs')
    old=read(previous/'source_identity.json')['sources']
    for n,h in old.items():
        if n in ALLOWED_OLD:
            if h!=ALLOWED_OLD[n]:raise ValueError('Unrecognized failed release')
        elif proof['sources'].get(n)!=h:raise ValueError('Model/loss/dependency source changed '+n)
    if read(previous/'ENGINEERING_PASSED.json')['status']!='PASS':raise ValueError('No passed engineering')
    if not all(n in index for n in BASE):raise ValueError('Missing immutable baseline/engineering files')
    events={};files=list(BASE)
    for arm in ARMS:
        events[arm]=[]
        for epoch in range(1,25):
            names=[arm+'/head_epoch_%02d.pth'%epoch,arm+'/epoch_%02d_val.json'%epoch,arm+'/epoch_%02d_val.rows.jsonl.gz'%epoch,
                   arm+'/epoch_%02d_train.json'%epoch,arm+'/epoch_%02d_train.rows.jsonl.gz'%epoch]
            present=[n in index for n in names]
            if not all(present):
                if any(present):raise ValueError('Partial epoch artifacts; do not silently retrain')
                if any(n.startswith(arm+'/epoch_') or n.startswith(arm+'/head_epoch_') for n in index if n not in files):raise ValueError('Gapped epoch prefix')
                break
            ep=read(previous/names[1])
            if ep['epoch']!=epoch or ep['checkpoint_sha256']!=index[names[0]]:raise ValueError('Epoch/weight identity differs')
            events[arm].append(ep);files+=names
    if not events[ARMS[0]] and not events[ARMS[1]]:raise ValueError('No completed epoch to recover')
    if events[ARMS[1]] and len(events[ARMS[0]])!=24:raise ValueError('Unexpected arm ordering')
    for name in files:
        target=out/name
        if target.exists():raise ValueError('Recovery cannot overwrite files '+name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(previous/name,target)
        if sha(target)!=index[name]:raise ValueError('Recovery copy mismatch')
    return dict(previous_directory=str(previous),failed_artifacts_sha256=sha(previous/'artifacts.json'),
        failed_finished=finished,previous_input_identity=read(previous/'input_identity.json'),
        previous_source_identity_sha256=sha(previous/'source_identity.json'),copied_files={n:index[n] for n in files},
        completed_epochs={a:len(v) for a,v in events.items()},events=events,
        completed_updates=sum(len(v)*639 for v in events.values()),engineering_reused=True,
        boundary='I/O-only repair. No learning/selection change; no repeated completed epoch, no discarded formal update. Model/optimizer/RNG restored from last complete checkpoint; remaining arm starts fresh.')
