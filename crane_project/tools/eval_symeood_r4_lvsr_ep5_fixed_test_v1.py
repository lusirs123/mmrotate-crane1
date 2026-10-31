"""One fixed descriptive R4/M20/L5 TEST; no training, reselection or depth.

The LVSR training selection remains null. The user authorized its predeclared
VAL representative ep5 for this single TEST. Static checks do not import torch.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
from crane_project.tools import eval_symeood_r4_midpoint_fixed_test_v1 as fixed

VERSION = 'symeood_r4_m20_l5_fixed_descriptive_test_v1'
TEST_ROOT = p.BASE/'lvsr/test'
OUT = TEST_ROOT/'attempt_002'
PROTOCOL = TEST_ROOT/'protocol_attempt_002.json'
HEAD_SHA = 'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e'
PINS = {
    'selection.json': '94bfe79882ae625863639c64d09df07209cf03b4b2d11600b84649166eda7887',
    'completion.json': 'c82784dc48f0fcf20db05d721f6313cdf4bb1b6b3127e496ccd55a01c9af958b',
    'epoch_05_VAL.rows.jsonl.gz': '06fffb95f43304a6e3a88d5942cdee8bc97ecb1298edb939cbd677eecf448f56',
    'head_epoch_05.pth': HEAD_SHA,
}
EXTRA_SOURCES = (
    'crane_project/data_contracts/symeood_r4_lvsr_pair_v1.json',
    'crane_project/tools/eval_symeood_r4_midpoint_fixed_test_v1.py',
    'crane_project/tools/eval_symeood_r4_lvsr_ep5_fixed_test_v1.py',
    'tests/test_symeood_r4_lvsr_ep5_fixed_test_v1.py',
)
read, save = fixed.read, fixed.save


def training_directory():
    for folder in (p.BASE/'lvsr/attempt_001/train',
                   p.BASE/'readbacks/lvsr_completed/lvsr/attempt_001/train'):
        if (folder/'selection.json').exists():
            for name, digest in PINS.items():
                if p.sha(folder/name) != digest:
                    raise ValueError('Fixed LVSR artifact differs: '+name)
            return folder
    raise ValueError('Completed R4/M20 LVSR artifacts required')


def validate_selection(selection, complete, identity):
    expected = dict(path='head_epoch_05.pth', sha256=HEAD_SHA, epoch=5)
    if (selection['identity'] != identity or complete['identity'] != identity
            or selection['protocol'] != p.VERSION or selection['gate'] != p.GATE
            or selection['selected_epoch'] is not None or selection['selected_checkpoint'] is not None
            or selection['representative_epoch'] != 5 or selection['representative_checkpoint'] != expected
            or selection['eligible_epochs'] != [] or len(selection['epochs']) != 24
            or selection['selection_on_test'] is not False or selection['automatic_adoption'] is not False
            or selection['TEST'] is not False or selection['depth'] is not False
            or complete['status'] != 'R4_M20_LVSR_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED'
            or complete['epochs_completed'] != 24 or complete['updates'] != 15336
            or complete['selected_epoch'] is not None or complete['representative_epoch'] != 5
            or complete['detector_updates'] != 0 or complete['midpoint_updates'] != 0
            or complete['TEST'] is not False or complete['depth'] is not False):
        raise ValueError('Requires completed fresh LVSR and unchanged null selection / representative5')
    if p.choose(selection['epochs']) != {k:selection[k] for k in
            ('selected_epoch', 'representative_epoch', 'eligible_epochs', 'selection_on_test', 'automatic_adoption')}:
        raise ValueError('Original VAL representative cannot be reproduced')


def validate_payload(payload, identity):
    if (payload['protocol'] != p.VERSION or payload['identity'] != identity
            or payload['epoch'] != 5 or payload['updates'] != 3195 or payload['updates_per_epoch'] != 639
            or payload['parameters'] != 2480 or payload['parent_checkpoint'] is not None
            or payload['detector_sha256'] != p.parent.R4_SHA
            or payload['midpoint_sha256'] != p.MIDPOINT_SHA):
        raise ValueError('Requires frozen fresh-trained R4/M20 LVSR ep5 payload')


def static_identity():
    identity = p.checked_sources()
    folder = training_directory()
    validate_selection(read(folder/'selection.json'), read(folder/'completion.json'), identity)
    return dict(protocol=VERSION, training_identity=identity, training_artifacts=PINS,
        midpoint_identity=fixed.static_identity(),
        sources={name:p.sha(ROOT/name) for name in EXTRA_SOURCES},
        detector_sha256=p.parent.R4_SHA, midpoint_sha256=p.MIDPOINT_SHA, lvsr_sha256=HEAD_SHA,
        detector_epoch=16, midpoint_epoch=20, lvsr_epoch=5, selected_epoch=None,
        representative_only=True, physical_gpu=2, val_names=list(fixed.VAL_NAMES),
        test_frames=1440, test_counts=fixed.COUNTS, memory=p.parent.MEMORY,
        training_updates=0, optimizer_construction=False, selection_on_test=False,
        checkpoint_scan=False, threshold_changes=False, cache_rebuilt=False,
        depth=False, exposed_TEST=True, automatic_adoption=False,
        baseline_policy='Reuse saved EOOD; replay R4/M only within the single required LVSR feature pass')


def sample_references(rows):
    indexed = {r['image']:r for r in rows}
    if len(rows) != 887 or len(indexed) != 887:
        raise ValueError('Full fixed epoch5 VAL rows required')
    return [indexed[n] for n in fixed.VAL_NAMES]


def compare_boxes(actual, reference, arms):
    import numpy as np
    maxima = {}
    for arm in arms:
        a, b = actual[arm], reference[arm]
        if ((a is None) != (b is None) or a is not None and
                (a[5] != b[5] or not np.allclose(a[:5], b[:5], atol=1e-4, rtol=1e-6))):
            raise ValueError('Frozen online replay differs: '+actual.get('image', '?')+'/'+arm)
        maxima[arm] = max(abs(x-y) for x,y in zip(a[:5],b[:5])) if a is not None else 0.
    return maxima


def compare_decisions(actual, reference):
    for key in ('size_delivery', 'failed_checks', 'vector_valid'):
        if actual.get(key) != reference.get(key):
            raise ValueError('LVSR VAL decision differs: '+key)


def prepare():
    identity = static_identity()
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    save(PROTOCOL, dict(identity, user_authorized=True,
        authorized_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    print('FIXED_L5_TEST_PREPARED_NO_GPU_OR_NEW_TEST_ACCESS', flush=True)


def infer():
    lock = read(PROTOCOL); identity = static_identity()
    if ({k:lock[k] for k in identity} != identity or not lock['user_authorized'] or OUT.exists()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('WORLD_SIZE','1') != '1'):
        raise ValueError('Fresh output, frozen protocol and single authorized GPU2 required')
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    adapted.resource_check(2, before=True)
    engine = adapted.make_engine(2)
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as legacy
    from crane_project.tools.eval_port_geometry_ap_v1 import validate_saved_gt
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    import numpy as np
    from PIL import Image
    torch, old, g = engine.torch, engine.old, engine.g
    original_optimizer_init = torch.optim.Optimizer.__init__
    def forbidden(*args, **kwargs):
        raise AssertionError('No optimizer during frozen TEST')
    torch.optim.Optimizer.__init__ = forbidden
    OUT.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    report = dict(protocol=VERSION, training_updates=0, selection_on_test=False,
        selected_epoch=None, representative_epoch=5, lvsr_sha256=HEAD_SHA,
        detector_sha256=p.parent.R4_SHA, midpoint_sha256=p.MIDPOINT_SHA,
        test_access=False, depth=False, automatic_adoption=False, physical_gpu=2)
    resources=[]; handle=None
    def status(stage, **details):
        tmp=OUT/'status.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), physical_gpu=2,
            time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), **details), indent=2))
        tmp.replace(OUT/'status.json')
    def resource():
        row=engine.resource_check(0); resources.append(row)
        with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    try:
        save(OUT/'frozen_contract.json', lock)
        runtime=engine.runtime(0); engine.seed_all(); torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        folder=training_directory()
        detector, mid, cfg=engine.load_selected_pipeline(fixed.training_directory()/'train/selection.json',0)
        if runtime != read(fixed.training_directory()/'roi_cache/cache_manifest.json')['runtime']:
            raise ValueError('Frozen training runtime differs')
        payload=torch.load(str(folder/'head_epoch_05.pth'),map_location='cpu')
        validate_payload(payload,identity['training_identity'])
        head=PairSpecificSizeHead().cuda(); head.load_state_dict(payload['head_state'],strict=True)
        for model in (detector,mid,head):model.eval().requires_grad_(False)
        def states():
            return {n:g.state_digest(m) for n,m in [('detector',detector),('midpoint',mid),('lvsr',head)]}
        before=states()
        if before['lvsr'] != payload['head_digest']:
            raise ValueError('LVSR payload/state digest differs')
        captured={}; counts=dict(feature_extractions=0,native_head_calls=0,size_head_calls=0)
        def hook(module,inputs,output):
            captured.update(inputs=inputs,midpoint=output['boxes_original'])
        handle=mid.register_forward_hook(hook)
        def audit(event):
            if event['stage']=='test_feature_extracted':counts['feature_extractions']+=1
            elif event['stage']=='test_native_head_called':counts['native_head_calls']+=1
            else:raise ValueError('Unknown capture event')
        def predict(image, metas):
            captured.clear(); result=old.capture(detector,mid,image,metas,audit=audit)
            m=captured['midpoint']; value=head(*captured['inputs'],m)
            counts['size_head_calls']+=1
            result.update(edge_residual=value['boxes_original'][0].cpu().tolist() if len(m) else None,
                size_delivery='no_output' if not len(m) else 'neutral' if bool(value['neutral_bypass'][0]) else
                    'square' if bool(value['square_bypass'][0]) else 'accepted' if bool(value['accepted'][0]) else 'fallback',
                size_failed_checks=[k for k,v in value['checks'].items() if len(m) and not bool(v[0])],
                vector_valid=bool(value['vector_valid'][0]) if len(m) else None,
                delta_roi=value['delta_roi'][0].cpu().tolist() if len(m) else None)
            if not frozen_identity(result):raise ValueError('LVSR changed M center/angle/score/presence/axis')
            return result
        status('sample_VAL_gate',done=0,total=8)
        with gzip.open(folder/'epoch_05_VAL.rows.jsonl.gz','rt') as f:
            samples=sample_references([json.loads(v) for v in f])
        manifest=read(g.ready.DATA/'manifest.json')
        metadata={r['id']:r for r in manifest['records'] if r['split']=='val'}
        dataset=engine.dataset_for(cfg,'val',1.)
        lookup={Path(r['filename']).stem:i for i,r in enumerate(engine.infos(dataset))}
        fresh=[]; checks=[]
        with torch.no_grad():
            for ref in samples:
                meta=metadata[ref['image']]
                for key in ('images','annfiles'):
                    if p.sha(g.ready.DATA/meta[key]['path']) != meta[key]['sha256']:
                        raise ValueError('Fixed VAL file SHA differs')
                with Image.open(g.ready.DATA/meta['images']['path']) as im:source=dict(ref,image_size=list(im.size))
                image,metas=old.load_view(dataset,lookup[ref['image']],source,0)
                pred=predict(image,metas)
                maximum=compare_boxes(dict(pred,image=ref['image']),ref,('b','midpoint','edge_residual'))
                compare_decisions(dict(pred,failed_checks=pred['size_failed_checks']),ref)
                checks.append(dict(image=ref['image'],max_box_abs=maximum))
                fresh.append(legacy.e.prediction_row(source,pred));resource()
        mrows=legacy.read_rows(fixed.training_directory()/'train/val_epoch_20.rows.jsonl')
        mrefs=fixed.representative_rows(mrows)
        with (OUT/'sample_M_VAL_reference.rows.jsonl').open('x') as f:
            for r in mrefs:f.write(json.dumps(r)+'\n')
        replay=legacy.e.verify_val_rows(fresh,OUT/'sample_M_VAL_reference.rows.jsonl')
        if states()!=before:raise ValueError('VAL gate changed frozen model state')
        save(OUT/'sample_VAL_gate.json',dict(passed=True,frames=8,updates=0,full_VAL_replayed=False,
            rows=checks,midpoint_replay=replay))
        del dataset, mrows, mrefs, samples, fresh, payload
        # TEST is opened only after the fixed online/cache gate passes.
        status('TEST_identity'); report['test_access']=True
        sources,data_identity=old.fixed_test_inputs()
        if p.sha(ROOT/fixed.TEST_REFERENCE)!=fixed.TEST_REFERENCE_SHA:
            raise ValueError('Common sealed TEST reference SHA differs')
        refs=legacy.read_rows(ROOT/fixed.TEST_REFERENCE)
        previous=legacy.read_rows(fixed.OUT/'test_rows.jsonl')
        previous_complete=read(fixed.OUT/'completion.json')
        if (previous_complete['status']!='FIXED_R4_M_EP20_TEST_COMPLETE' or
                previous_complete['head_sha256']!=p.MIDPOINT_SHA or previous_complete['detector_sha256']!=p.parent.R4_SHA
                or len(sources)!=1440 or len(previous)!=1440 or len(refs)!=1440
                or [r['image'] for r in sources]!=[r['image'] for r in previous]
                or [r['image'] for r in sources]!=[r['image'] for r in refs]):
            raise ValueError('Fixed full TEST/R4/M baseline identity differs')
        previous_sha=p.sha(fixed.OUT/'test_rows.jsonl')
        for source,ref in zip(sources,refs):
            ann=g.ready.DATA/'test/annfiles'/(source['image']+'.txt')
            poly=np.asarray([float(v) for v in ann.read_text().split()[:8]]).reshape(4,2)
            validate_saved_gt(ref['gt'],poly,np.asarray(source['gt']));source['gt']=ref['gt']
        dataset=old.build_test_dataset(cfg,sources); maxima=dict(b=0.,midpoint=0.)
        with torch.no_grad(),(OUT/'test_rows.jsonl').open('x') as stream:
            for index,source in enumerate(sources):
                image,metas=old.load_view(dataset,index,source,0)
                pred=predict(image,metas)
                row=legacy.e.prediction_row(source,pred)
                row.update(split='test')
                if row['gt']!=previous[index]['gt']:raise ValueError('Same GT required')
                for key,value in compare_boxes(row,previous[index],('b','midpoint')).items():maxima[key]=max(maxima[key],value)
                stream.write(json.dumps(g.json_native(row),allow_nan=False)+'\n');stream.flush();resource()
                if index%50==0 or index==1439:
                    status('TEST',done=index+1,total=1440);print('FIXED R4+M20+L5 TEST',index+1,'/1440',flush=True)
        expected=1448
        if (states()!=before or counts!=dict(feature_extractions=expected,native_head_calls=3*expected,size_head_calls=expected)
                or static_identity()!=identity or p.sha(ROOT/p.parent.R4_CHECKPOINT)!=p.parent.R4_SHA
                or p.sha(fixed.OUT/'test_rows.jsonl')!=previous_sha):
            raise ValueError('Frozen state/source/artifact or one-pass call count differs')
        save(OUT/'existing_baseline_replay.json',dict(frames=1440,max_box_abs=maxima,scores_exact=True,
            baseline_rows_sha256=previous_sha,separate_baseline_forward=False))
        report.update(status='FIXED_R4_M20_L5_TEST_COMPLETE',frames=1440,
            domain_counts=dict(Counter(r['domain'] for r in sources)),runtime=runtime,counters=counts,
            state_before=before,state_after=states(),data_identity=data_identity,
            reference_gt_sha256=fixed.TEST_REFERENCE_SHA,elapsed_seconds=time.perf_counter()-started,
            resource=dict(own_peak_mib=max(r['own_process_mib'] for r in resources),
                minimum_free_mib=min(r['free_mib'] for r in resources),
                allocator_peak_mib=torch.cuda.max_memory_allocated(0)/2**20))
        save(OUT/'completion.json',report);status('complete',frames=1440,updates=0)
        save(OUT/'artifacts.json',dict(files={f.name:p.sha(f) for f in OUT.iterdir() if f.is_file()}))
        print('FIXED_R4_M20_L5_TEST_COMPLETE',flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),elapsed_seconds=time.perf_counter()-started)
        save(OUT/'failure.json',report);status('stopped_on_error',error=repr(error),retry=False)
        raise
    finally:
        if handle is not None:handle.remove()
        torch.optim.Optimizer.__init__=original_optimizer_init


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage',choices=('check','prepare','infer'),default='check')
    return result


if __name__=='__main__':
    args=parser().parse_args()
    if args.stage=='check':print(json.dumps(static_identity(),indent=2))
    else:globals()[args.stage]()
