#!/usr/bin/env python3
"""One frozen exploratory ep11 TEST and depth evaluation using existing routines."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import read, write, sha
from crane_project.tools import eval_symeood_110_lvsr_downstream as existing

CONTRACT = ROOT/'crane_project/data_contracts/symeood_110_lvsr_ep11_test_v1.json'


def checked():
    lock = read(CONTRACT)
    for name, digest in lock['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source differs: '+name)
    for name, digest in lock['artifacts'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Artifact differs: '+name)
    # Reuse the original closure, frontend and midpoint identity checks.
    existing.checked_sources()
    return lock


def detection_report(predictions):
    """Bind only the evaluated arm for the unchanged generic report routine."""
    original = existing.HEADS
    try:
        existing.HEADS = {'ep11': {}}
        return existing.detection_report(predictions)
    finally:
        existing.HEADS = original


def depth_report(predictions, truths, manifest, calibration):
    original_heads, original_arms = existing.HEADS, existing.ARMS
    try:
        existing.HEADS = {'ep11': {}}
        existing.ARMS = ('b', 'midpoint', 'ep11')
        return existing.depth_report(predictions, truths, manifest, calibration)
    finally:
        existing.HEADS, existing.ARMS = original_heads, original_arms


def run():
    lock = checked()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(lock['physical_gpu']):
        raise ValueError('Use the pinned single physical GPU')
    out = ROOT/lock['output_directory']
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    def status(stage, **values):
        temp = out/'status.tmp'
        temp.write_text(json.dumps(dict(stage=stage, pid=os.getpid(),
            time=datetime.now(timezone.utc).isoformat(), **values), indent=2))
        temp.replace(out/'status.json')
    try:
        from crane_project.tools import train_symeood_110_midpoint_v1 as f
        f.runtime_imports()
        torch = f.torch
        from crane_project.tools import eval_port_geometry_midpoint_v1_test as helper
        from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
        from mmdet.datasets.pipelines import Compose
        from PIL import Image
        runtime = f.runtime(0)
        if runtime != read(existing.FORMAL/'roi_cache/cache_manifest.json')['runtime']:
            raise ValueError('Runtime differs from the frozen chain')
        f.seed_all()
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        detector, midpoint, cfg = f.load_selected_pipeline(existing.FORMAL/'train/selection.json', 0)
        pipeline = Compose(deepcopy(cfg.data.val.pipeline))
        payload = torch.load(str(ROOT/lock['checkpoint']), map_location='cpu')
        if payload['epoch'] != 11 or payload['arm'] != 'fixed_b':
            raise ValueError('Requires the original common-stage ep11')
        if (payload['input_identity']['detector_sha256'] != existing.training.base.DETECTOR_SHA or
                payload['input_identity']['midpoint_sha256'] != existing.training.base.MIDPOINT_SHA):
            raise ValueError('Head frontend identity differs')
        head = AlignedSizeHead('fixed_b').cuda()
        head.load_state_dict(payload['head_state'], strict=True)
        for model in (detector, midpoint, head):
            model.eval().requires_grad_(False)
        def states():
            return {name: f.g.state_digest(model) for name, model in
                    [('detector', detector), ('midpoint', midpoint), ('ep11', head)]}
        before = states()
        captured = {}
        def hook(module, inputs, output):
            captured.update(inputs=inputs, midpoint=output['boxes_original'])
        handle = midpoint.register_forward_hook(hook)
        counts = dict(feature_extractions=0, native_head_calls=0, size_head_calls=0)
        def audit(event):
            key = 'feature_extractions' if event['stage']=='test_feature_extracted' else 'native_head_calls'
            counts[key] += 1
        def predict(source):
            image, metas = existing.native_view(pipeline, source, helper)
            captured.clear()
            result = helper.capture(detector, midpoint, image, metas, audit)
            roi, support, b, bm, xy = captured['inputs']
            m = captured['midpoint']
            value = head(roi, support, b, bm, xy, m)
            counts['size_head_calls'] += 1
            result['ep11'] = value['boxes_original'][0].cpu().tolist() if len(m) else None
            result['delivery'] = {'ep11': existing.delivery_label(value)}
            result['failed_checks'] = {'ep11': [k for k,v in value['checks'].items() if not bool(v[0])] if len(m) else []}
            result['vector_valid'] = {'ep11': bool(value['vector_valid'][0]) if len(m) else None}
            existing.identity(result['midpoint'], result['ep11'])
            return result
        write(out/'frozen_contract.json', lock)
        status('sample_VAL_gate')
        with gzip.open(ROOT/lock['val_rows'], 'rt') as stream:
            refs = {r['image']: r for r in map(json.loads, stream)}
        if len(refs) != 887:
            raise ValueError('Saved VAL identity differs')
        sample = []
        with torch.no_grad():
            for name in lock['val_names']:
                ref = refs[name]
                source = dict(ref, image_path=str(f.g.ready.DATA/'val/images'/(name+'.jpg')))
                with Image.open(source['image_path']) as image:
                    source['image_size'] = list(image.size)
                predicted = predict(source)
                reference = dict(ref, ep11=ref['edge_residual'])
                maximum = existing.compare_frame(predicted, reference, ('b','midpoint','ep11'))
                for key, expected in [('delivery',ref['size_delivery']),
                        ('failed_checks',ref['failed_checks']), ('vector_valid',ref.get('vector_valid'))]:
                    if predicted[key]['ep11'] != expected:
                        raise ValueError('VAL decision differs: '+key)
                sample.append(dict(image=name, maximum_box_difference=maximum))
        write(out/'sample_VAL_gate.json', dict(passed=True, frames=len(sample), rows=sample))
        sources, data = helper.fixed_test_inputs()
        if len(sources) != 1440:
            raise ValueError('Fixed TEST frame count differs')
        saved = {r['image']:r for r in existing.rows(existing.OUT/'test_rows.jsonl')}
        predictions = []
        maxima = dict(b=0., midpoint=0.)
        status('TEST', done=0, total=1440)
        with torch.no_grad(), (out/'test_rows.jsonl').open('x') as stream:
            for i, source in enumerate(sources):
                source.update(image_path=str(f.g.ready.DATA/'test/images'/(source['image']+'.jpg')))
                result = predict(source)
                result.update(**{k:source[k] for k in ('image','sequence','domain','frame_id')},
                    scale=1., split='test', gt=f.np.asarray(source['gt'], dtype=f.np.float32).tolist())
                differences = existing.compare_frame(result, saved[source['image']], ('b','midpoint'))
                for key, value in differences.items():
                    maxima[key] = max(maxima[key], value)
                predictions.append(result)
                stream.write(json.dumps(result, allow_nan=False)+'\n')
                stream.flush()
                if (i+1)%100 == 0 or i+1 == 1440:
                    status('TEST', done=i+1, total=1440)
                    f.resource_check(0)
                    print('TEST', i+1, '/1440', flush=True)
        write(out/'test_summary.json', detection_report(predictions))
        write(out/'test_input_identity.json', data)
        write(out/'existing_TEST_baseline_replay.json', dict(frames=1440, maximum_box_difference=maxima))
        from tools import eval_port_frozen_downstream_v1 as legacy
        frozen = read(ROOT/'tools/port_frozen_downstream_v1_protocol.json')
        pin = frozen['sequences'][existing.SEQUENCE]
        manifest, truth_path, images = legacy.checked_dataset(ROOT/'crane_project/data/webots_depth', existing.SEQUENCE, pin)
        if len(images) != 1097:
            raise ValueError('Fixed depth sequence count differs')
        calibration = read(ROOT/existing.depth.CALIBRATION)
        baseline_depth = {r['frame_id']:r for r in existing.rows(existing.OUT/'depth_predictions.jsonl')}
        depth_predictions = []
        maxima = dict(b=0., midpoint=0.)
        status('Unknown02', done=0, total=1097)
        with torch.no_grad(), (out/'depth_predictions.jsonl').open('x') as stream:
            for i, source in enumerate(images):
                if sha(source['image_path']) != source['sha256']:
                    raise ValueError('Depth image identity differs')
                result = predict(source)
                result.update(frame_id=source['frame_id'], image_sha256=source['sha256'])
                differences = existing.compare_frame(result, baseline_depth[source['frame_id']], ('b','midpoint'))
                for key, value in differences.items():
                    maxima[key] = max(maxima[key], value)
                depth_predictions.append(result)
                stream.write(json.dumps(result, allow_nan=False)+'\n')
                stream.flush()
                if (i+1)%100 == 0 or i+1 == len(images):
                    status('Unknown02', done=i+1, total=1097)
                    f.resource_check(0)
                    print('Unknown02', i+1, '/1097', flush=True)
        truths = existing.rows(truth_path)
        legacy.validate_truth(truths, manifest, pin)
        evaluated, summary = depth_report(depth_predictions, truths, manifest, calibration)
        existing.write_rows(out/'depth_rows.jsonl', evaluated)
        write(out/'depth_summary.json', summary)
        write(out/'depth_input_identity.json', dict(pin=pin, camera=manifest['camera'],
            physical_reference=manifest['obb_reference_geometry'], calibration_sha256=existing.depth.CAL_SHA))
        write(out/'existing_depth_baseline_replay.json', dict(frames=1097, maximum_box_difference=maxima))
        handle.remove()
        if states() != before or checked() != lock:
            raise ValueError('Frozen states or inputs changed')
        expected = 1440+1097+len(sample)
        if counts != dict(feature_extractions=expected, native_head_calls=3*expected, size_head_calls=expected):
            raise ValueError('Unexpected inference count')
        write(out/'completion.json', dict(state='FROZEN_EXPLORATORY_EP11_TEST_DEPTH_COMPLETE',
            frames=1440, depth_frames=1097, checkpoint_sha256=lock['checkpoint_sha256'], physical_gpu=lock['physical_gpu'],
            runtime=runtime, counters=counts, training_updates=0, depth=True, selection=False,
            original_eligibility_unchanged=True, elapsed_seconds=time.monotonic()-started,
            state_before=before, state_after=states(), resource=f.resource_check(0)))
        status('complete', done=1440, total=1440)
        write(out/'artifacts.json', dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    except BaseException as error:
        write(out/'failure.json', dict(error=repr(error), training_updates=0, depth=True))
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
    else:
        print(json.dumps(dict(checked=True, protocol=checked()['protocol'], updates=0)))
