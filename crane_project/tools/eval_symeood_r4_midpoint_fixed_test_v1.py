"""One authorized fixed R4 ep16 + original VAL-selected M ep20 TEST.

Static preparation imports no torch and opens no TEST. Inference first replays
eight fixed VAL rows, then uses the sealed native capture once per TEST frame.
No cache rebuild, optimizer, checkpoint scan, threshold override or LVSR.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import symeood_r4_midpoint_protocol_v1 as p

VERSION = 'symeood_r4_ep16_midpoint_ep20_fixed_test_v1'
BASE = p.DEFAULT_ROOT
TEST_ROOT = BASE/'test'
OUT = TEST_ROOT/'attempt_001'
HEAD_SHA = 'd595ab01c3aea48aec66d6cf1e38460990ea272bb5ca962ca8ec8e17dd8d6dce'
ARTIFACTS = {
    'train/selection.json': 'ade312204703c975b4c95f0eae3f9fe45377fd83140b0182d13746938bd2966a',
    'train/completion.json': '6540a2fbb6d2c6163c0335712751c580cc8d455841e69d1cfbce4d615facd240',
    'train/val_epoch_20.rows.jsonl': 'f7762c38b798b72683452da9faf5a390025e176c2ba9ddc6c49bf1a6b056979f',
    'roi_cache/cache_manifest.json': '7034d4073c16538d5b1683753934a31c2a108f39e9183b217d850f6762eb8017',
    'train/head_epoch_20.pth': HEAD_SHA,
}
VAL_NAMES = ('real_seq07_00001', 'real_seq07_00226', 'real_seq14_00000',
    'real_seq14_00148', 'sim_seq10_00000', 'sim_seq10_00511',
    'real_seq07_00193', 'real_seq07_00020')
TEST_REFERENCE = 'work_dirs/smallpaper_readiness_20261028_v1/saved_predictions_unified.rows.jsonl'
TEST_REFERENCE_SHA = '1abae6afe6ece14c356c7b5608a5ceb692316172605d970444c41a2a12fa8481'
PREVIOUS_R4 = 'work_dirs/symeood_110_reg_tower_20261030_v1/fixed_TEST_20261030_v1'
COUNTS = dict(real_seq03=200, real_seq04=668, sim_seq09=572)
EXTRA_SOURCES = ('crane_project/data_contracts/symeood_r4_midpoint_v1.json',
    'crane_project/tools/eval_symeood_r4_midpoint_fixed_test_v1.py',
    'tests/test_symeood_r4_midpoint_fixed_test_v1.py')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def training_directory():
    for folder in (BASE/'midpoint/attempt_001', BASE/'readbacks/completed/midpoint/attempt_001'):
        if (folder/'train/selection.json').exists():
            for name, digest in ARTIFACTS.items():
                if p.sha(folder/name) != digest:
                    raise ValueError('Fixed R4/M artifact differs: '+name)
            return folder
    raise ValueError('Completed R4 midpoint artifacts required')


def validate_selection(selection, complete, identity):
    cp = selection['selected_checkpoint']
    if (selection['protocol'] != p.VERSION or selection['identity'] != identity
            or selection['split'] != 'val' or selection['test_access'] is not False
            or selection['selection_on_test'] is not False
            or cp['epoch'] != 20 or cp['path'] != 'head_epoch_20.pth' or cp['sha256'] != HEAD_SHA
            or cp['updates'] != 17980 or len(selection['all_checkpoints']) != 24
            or complete['epochs_completed'] != 24 or complete['updates'] != 21576
            or complete['detector_updates'] != 0 or complete['test_access'] is not False
            or complete['selection'] != selection):
        raise ValueError('Original full-VAL-selected R4/M identity differs')


def static_identity():
    identity = p.checked_sources()
    directory = training_directory()
    validate_selection(read(directory/'train/selection.json'), read(directory/'train/completion.json'), identity)
    return dict(protocol=VERSION, training_identity=identity, training_artifacts=ARTIFACTS,
        sources={name:p.sha(ROOT/name) for name in EXTRA_SOURCES},
        detector_sha256=p.R4_SHA, head_sha256=HEAD_SHA, selected_head_epoch=20,
        physical_gpu=2, val_names=list(VAL_NAMES), test_frames=1440, test_counts=COUNTS,
        reference_rows=TEST_REFERENCE, reference_sha256=TEST_REFERENCE_SHA,
        memory=p.MEMORY, detector_updates=0, head_updates=0, optimizer_construction=False,
        checkpoint_scan=False, threshold_changes=False, selection_on_test=False,
        cache_rebuilt=False, LVSR=False, test_repeatedly_exposed=True, automatic_adoption=False)


def representative_rows(rows):
    indexed = {r['image']:r for r in rows}
    if len(rows) != 887 or len(indexed) != 887:
        raise ValueError('Full selected VAL rows required')
    selected = [indexed[name] for name in VAL_NAMES]
    if ({r['domain'] for r in selected} != {'real', 'sim'}
            or {r['image'] for r in selected if r['accepted'] is False} != {'real_seq07_00193'}):
        raise ValueError('VAL representatives must preserve the saved fallback case')
    return selected


def prepare():
    identity = static_identity()
    TEST_ROOT.mkdir(parents=True, exist_ok=False)
    save(TEST_ROOT/'protocol.json', dict(identity,
        authorized_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        user_authorized=True, stage='FROZEN_BEFORE_TEST_INFERENCE'))
    print('R4_M_FIXED_TEST_PREPARED_NO_GPU_OR_TEST_ACCESS', flush=True)


def infer():
    lock = read(TEST_ROOT/'protocol.json')
    identity = static_identity()
    if {k:lock[k] for k in identity} != identity or not lock['user_authorized']:
        raise ValueError('Prepared TEST identity differs')
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('Fresh fixed output and authorized physical GPU2 required')
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    adapted.resource_check(2, before=True)
    engine = adapted.make_engine(2)
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as legacy
    from crane_project.tools.eval_port_geometry_ap_v1 import validate_saved_gt
    import numpy as np
    from PIL import Image
    e, old, g, torch = legacy.e, engine.old, engine.g, engine.torch
    original_optimizer_init = torch.optim.Optimizer.__init__
    def forbidden(*args, **kwargs):
        raise AssertionError('No optimizer may be constructed during frozen TEST')
    torch.optim.Optimizer.__init__ = forbidden
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(protocol=VERSION, detector_sha256=p.R4_SHA, head_sha256=HEAD_SHA,
        selected_head_epoch=20, physical_gpu=2, detector_updates=0, head_updates=0,
        test_access=False, selection_on_test=False, test_repeatedly_exposed=True,
        feature_extractions=0, native_head_calls=0, cache_rebuilt=False, LVSR=False,
        automatic_adoption=False)
    resources = []
    def status(stage, **details):
        value = dict(stage=stage, time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            pid=os.getpid(), physical_gpu=2, **details)
        tmp=OUT/'status.tmp'; tmp.write_text(json.dumps(value, indent=2)); tmp.replace(OUT/'status.json')
    def resource():
        row = engine.resource_check(0); resources.append(row)
        with (OUT/'resources.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), **row))+'\n')
    try:
        save(OUT/'frozen_contract.json', lock)
        runtime = engine.runtime(0); engine.seed_all()
        directory = training_directory()
        detector, head, cfg = engine.load_selected_pipeline(directory/'train/selection.json', 0)
        cache = read(directory/'roi_cache/cache_manifest.json')
        if runtime != cache['runtime'] or p.sha(ROOT/p.R4_CHECKPOINT) != p.R4_SHA:
            raise ValueError('Frozen runtime/checkpoint differs')
        before = dict(detector=g.state_digest(detector), head=g.state_digest(head))
        saved = legacy.read_rows(directory/'train/val_epoch_20.rows.jsonl')
        samples = representative_rows(saved)
        status('sample_VAL_gate', done=0, total=8)
        manifest = read(g.ready.DATA/'manifest.json')
        metadata = {r['id']:r for r in manifest['records'] if r['split']=='val'}
        dataset = engine.dataset_for(cfg, 'val', 1.)
        lookup = {Path(r['filename']).stem:i for i,r in enumerate(engine.infos(dataset))}
        fresh = []
        with (OUT/'sample_VAL_reference.rows.jsonl').open('x') as f:
            for r in samples: f.write(json.dumps(r, allow_nan=False)+'\n')
        with torch.no_grad():
            for r in samples:
                meta = metadata[r['image']]
                for key in ('images', 'annfiles'):
                    if p.sha(g.ready.DATA/meta[key]['path']) != meta[key]['sha256']:
                        raise ValueError('Representative VAL file SHA differs')
                with Image.open(g.ready.DATA/meta['images']['path']) as im:
                    source=dict(r, image_size=list(im.size))
                image, metas = old.load_view(dataset, lookup[r['image']], source, 0)
                fresh.append(e.prediction_row(source, old.capture(detector, head, image, metas)))
                resource()
        replay = e.verify_val_rows(fresh, OUT/'sample_VAL_reference.rows.jsonl')
        if dict(detector=g.state_digest(detector), head=g.state_digest(head)) != before:
            raise ValueError('VAL gate changed model state')
        save(OUT/'sample_VAL_gate.json', dict(passed=True, names=list(VAL_NAMES), updates=0,
            full_VAL_replayed=False, **replay))
        del dataset, saved, samples, fresh
        # Fixed full TEST access begins only after the same-weight VAL gate.
        status('TEST_identity')
        report['test_access'] = True
        sources, data_identity = old.fixed_test_inputs()
        reference_path=ROOT/TEST_REFERENCE
        if p.sha(reference_path) != TEST_REFERENCE_SHA:
            raise ValueError('Sealed common TEST GT reference differs')
        refs = legacy.read_rows(reference_path)
        if [r['image'] for r in refs] != [r['image'] for r in sources]:
            raise ValueError('Sealed/common TEST frame order differs')
        for source, ref in zip(sources, refs):
            ann=g.ready.DATA/'test/annfiles'/(source['image']+'.txt')
            poly=np.asarray([float(v) for v in ann.read_text().split()[:8]]).reshape(4,2)
            validate_saved_gt(ref['gt'], poly, np.asarray(source['gt']))
            source['gt']=ref['gt']  # Identical sealed GT for all compared methods; offline only.
        dataset=old.build_test_dataset(cfg, sources)
        def audit(event):
            if event['stage']=='test_feature_extracted': report['feature_extractions']+=1
            elif event['stage']=='test_native_head_called': report['native_head_calls']+=1
            else: raise ValueError('Unknown capture audit event')
        rows=[]
        with torch.no_grad(), (OUT/'test_rows.jsonl').open('x') as stream:
            for index, source in enumerate(sources):
                image, metas=old.load_view(dataset, index, source, 0)
                prediction=old.capture(detector, head, image, metas, audit=audit)
                row=e.prediction_row(source, prediction); rows.append(row)
                stream.write(json.dumps(g.json_native(row), allow_nan=False)+'\n'); stream.flush()
                resource()
                if index%50==0 or index==1439:
                    status('TEST', done=index+1, total=1440)
                    print('FIXED R4+M TEST', index+1, '/1440', flush=True)
        legacy.checked_rows(rows, 'test')
        after=dict(detector=g.state_digest(detector), head=g.state_digest(head))
        if (after != before or report['feature_extractions'] != 1440
                or report['native_head_calls'] != 4320 or static_identity() != identity
                or p.sha(ROOT/p.R4_CHECKPOINT) != p.R4_SHA):
            raise ValueError('Frozen state/source or one-pass capture count differs')
        value=dict(split='test', groups=old.test_summaries(rows))
        save(OUT/'test_compare.json', value)
        report.update(status='FIXED_R4_M_EP20_TEST_COMPLETE', frames=1440,
            domain_counts=dict(Counter(r['domain'] for r in rows)), runtime=runtime,
            data_identity=data_identity, reference_gt_sha256=TEST_REFERENCE_SHA,
            state_before=before, state_after=after, sample_VAL_frames=8,
            elapsed_seconds=time.perf_counter()-started,
            resource=dict(own_peak_mib=max(r['own_process_mib'] for r in resources),
                minimum_free_mib=min(r['free_mib'] for r in resources),
                allocator_peak_mib=torch.cuda.max_memory_allocated(0)/2**20))
        save(OUT/'completion.json', report)
        status('complete', frames=1440, updates=0)
        save(OUT/'artifacts.json', dict(files={f.name:p.sha(f) for f in OUT.iterdir() if f.is_file()}))
        print('FIXED_R4_M_EP20_TEST_COMPLETE', flush=True)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error), elapsed_seconds=time.perf_counter()-started)
        save(OUT/'failure.json', report)
        status('stopped_on_error', error=repr(error), retry=False)
        raise
    finally:
        torch.optim.Optimizer.__init__=original_optimizer_init


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', choices=('check', 'prepare', 'infer'), default='check')
    return result


if __name__=='__main__':
    args=parser().parse_args()
    if args.stage=='check': print(json.dumps(static_identity(), indent=2))
    else: globals()[args.stage]()
