#!/usr/bin/env python3
"""CPU-only support/annotation diagnosis from the sealed sigma1.5 collection.

No tensor, image, model, policy or Webots prediction is loaded. No fitting,
threshold choice, label rewrite, split reassignment or TEST evaluation.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_geometry_size_support_v1 as d

BASE = 'crane_project/tools/port_geometry_size_support_v1'
PROTOCOL = BASE+'_protocol.json'
SOURCES = BASE+'_sources.json'
LABELS = BASE+'_labels.json'
SAMPLES = 'crane_project/tools/port_geometry_g_v1_train_samples.json'
DATA = 'crane_project/data/crane_grab_port_day2night_v1'
SAMPLING = DATA+'/provenance/axis_k2p1/sampling_frame_sources.csv'
CONVERSION = DATA+'/provenance/axis_k2p1/conversion_manifest.json'
CALIBRATION = 'crane_project/configs/depth_scale_calibration_obb_raw_opt_short_q2_train03_v1.json'
DEFAULT_COLLECTION = 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1'
B_SHA = '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'
HEAD_SHA = '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7'
SELECTION_SHA = 'c52bf18854a514e171c4f5cf2e6afce69b7b372412f56f3d9ec74edadd9386c0'
ROWS_SHA = '0d971bb608761ecd9df180f3e6c0dffb46ca71974431cc97375b9cc9b14c88c6'
INPUT_FILES = {
    'collect/completion.json': '9af0d982326b20fef2a49b13b5e973ba1ba6e17101207f3abafdfda25981c9ac',
    'collect/collect_report.json': '3c39308d481e6154e9f23ba6055c52e8d7ed707994b262d0d8546c3cdf762922',
    'collect/input_check.json': 'b9679c4b2ec586e89992c411922acc9afa250a2e66ba969bb9c96daa14615318',
    'collect/final_rows.jsonl': ROWS_SHA,
    'collect/independent_B_cache_comparisons.json': '254525cd0a2530cfd738d342b0952fb33fc135519ad542ba96414af1e35e04c3'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def parse(data):
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(data, parse_constant=invalid)


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False)+'\n')


def relative(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Unsafe relative evidence path: '+name)
    return Path(name)


def protocol_document():
    return dict(protocol=d.VERSION, stages=['check', 'diagnose'], settings=d.SETTINGS,
        collection_artifacts=INPUT_FILES,
        frontend=dict(b_sha256=B_SHA, head_sha256=HEAD_SHA, sigma_cells=1.5, epoch=3,
                      updates=2706, selection_sha256=SELECTION_SHA),
        counts=d.COUNTS, sizes='Canonical final ORIGINAL pixel long/short edges; natural log pred/GT.',
        sampling='Standard scale only. Preserve legacy64 fit/probe, remaining TRAIN and all VAL roles.',
        runs='Break source video/segment/time gaps where mapped; other runs are frame-index-only. '
             'Overlapping flag runs are not independent scenes. Never automatically certify support sufficiency.',
        labels='Hash only fixed TRAIN/VAL annfiles and mapped axis snapshots. '
               'Polygon geometry is a rounding-aware diagnostic, not reconstructed native GT. '
               'Axis/k consistency does not prove Webots physical-reference matching.',
        scope=dict(model_loads=0, feature_extractions=0, parameter_updates=0,
            policy_loads=0, threshold_fits=0, depth_fits=0, test_prediction_reads=0,
            test_image_annotation_reads=0, webots_frame_reads=0, split_reassignments=0),
        promotion='Human review required; no architecture/loss weights/checkpoints selected or paper claims.')


def checked_sources(root=ROOT):
    manifest = parse((root/SOURCES).read_bytes())
    required = {PROTOCOL, LABELS, SAMPLES, SAMPLING, CONVERSION, CALIBRATION,
        'crane_project/utils/port_geometry_size_support_v1.py',
        'crane_project/tools/diagnose_port_geometry_size_support_v1.py',
        'crane_project/tools/run_port_geometry_size_support_v1.sh',
        'tests/test_port_geometry_size_support_v1.py'}
    if (manifest.get('protocol') != d.VERSION or set(manifest['sources']) != required or
            parse((root/PROTOCOL).read_bytes()) != protocol_document() or
            manifest['protocol_sha256'] != sha(root/PROTOCOL)):
        raise ValueError('Size support source/protocol closure differs')
    for name, expected in manifest['sources'].items():
        if sha(root/relative(name)) != expected:
            raise ValueError('Size support source SHA differs: '+name)
    labels = parse((root/LABELS).read_bytes())
    if (labels['protocol'] != d.VERSION or labels['test_records_included'] is not False or
            Counter(r['sequence'] for r in labels['records']) !=
            Counter(dict(d.COUNTS['train'], **d.COUNTS['val'])) or
            len({r['image'] for r in labels['records']}) != 3445):
        raise ValueError('TRAIN/VAL annotation contract differs')
    for r in labels['records']:
        path = relative(r['annfile'])
        if (r['split'] not in ('train', 'train_sim', 'val') or
                path.parts[:1] != (r['split'],) or path.parts[1:2] != ('annfiles',) or
                path.name != r['image']+'.txt'):
            raise ValueError('Non-allowlisted annotation path')
    return dict(protocol_sha256=sha(root/PROTOCOL), source_manifest_sha256=sha(root/SOURCES),
                sources=manifest['sources']), labels


def load_collection(directory=None, archive=None):
    """Consume only the five already-reviewed collect files; never extract tar."""
    if directory is not None and archive is not None:
        raise ValueError('Choose directory OR archive')
    if archive is not None:
        archive = Path(archive).resolve()
        before = sha(archive)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if len({m.name for m in members}) != len(members):
                raise ValueError('Duplicate archive member')
            for m in members:
                relative(m.name.rstrip('/'))
                if not (m.isdir() or m.isfile()) or m.size > 32*1024*1024:
                    raise ValueError('Unsafe or oversized archive member')
            if 'collect/failure.json' in {m.name for m in members}:
                raise ValueError('Failed collection cannot be used')
            blobs = {n: tar.extractfile(n).read() for n in INPUT_FILES}
        if sha(archive) != before:
            raise ValueError('Archive changed during read')
        location = dict(archive=str(archive), archive_sha256=before)
    else:
        directory = Path(directory or ROOT/DEFAULT_COLLECTION).resolve()
        if not (directory/'collect/completion.json').is_file():
            raise FileNotFoundError('Existing collection missing. Consult work_dirs/port_results/INDEX.md '
                'and pass --collection-dir RUN_PARENT or --collection-archive analysis_20261006.tar.gz. '
                'Do not rerun collection: '+str(directory))
        if (directory/'collect/failure.json').exists():
            raise ValueError('Failed collection cannot be used')
        if any((directory/n).stat().st_size > 32*1024*1024 for n in INPUT_FILES):
            raise ValueError('Oversized collection file')
        blobs = {n: (directory/n).read_bytes() for n in INPUT_FILES}
        if any(sha(directory/n) != digest(b) for n,b in blobs.items()):
            raise ValueError('Collection changed during read')
        location = dict(directory=str(directory))
    actual = {n:digest(b) for n,b in blobs.items()}
    if actual != INPUT_FILES:
        raise ValueError('Requires exact reviewed sigma1.5 collect bytes: '+
                         str([n for n in INPUT_FILES if actual[n] != INPUT_FILES[n]]))
    completion = parse(blobs['collect/completion.json'])
    contract = completion['contract']
    front = contract['front_end']; checkpoint = front['midpoint_checkpoint']
    report = parse(blobs['collect/collect_report.json'])
    if (completion['protocol'] != 'port_midpoint_sigma15_reliability_v1' or
            completion['status'] != 'SIGMA15_RELIABILITY_COLLECTION_COMPLETE' or
            front['frozen_b']['checkpoint_sha256'] != B_SHA or
            front['selection_sha256'] != SELECTION_SHA or checkpoint['sha256'] != HEAD_SHA or
            (checkpoint['epoch'], checkpoint['sigma_cells'], checkpoint['updates']) != (3,1.5,2706) or
            contract != parse(blobs['collect/input_check.json']) or
            completion['test_read'] is not False or contract['test_read'] is not False or
            any(completion[k] != 0 for k in ('detector_updates','midpoint_updates','reference_updates')) or
            report['head_state_before'] != report['head_state_after'] or
            report['head_state_after'] != checkpoint['head_digest'] or
            report['detector_loaded'] is not False or report['test_read'] is not False or
            any(report[k] != 0 for k in ('feature_extractions','head_updates','detector_updates'))):
        raise ValueError('Frozen frontend/collection proof differs')
    for name, expected in completion['artifacts'].items():
        if relative(name).name != name or 'collect/'+name not in actual or actual['collect/'+name] != expected:
            raise ValueError('Collection artifact receipt differs')
    rows = [parse(line) for line in blobs['collect/final_rows.jsonl'].splitlines() if line.strip()]
    d.checked_rows(rows)
    val = [dict(r) for r in rows if r['reliability_role'] == 'val']
    for r in val:
        r.pop('reliability_role')
    if fingerprint(val) != contract['final_VAL_fingerprint']:
        raise ValueError('VAL selected-epoch fingerprint differs')
    return rows, dict(location=location, artifacts=actual, front_end=front,
                      frozen_collection_report=report, final_VAL_fingerprint=fingerprint(val))


def sampling_map(root=ROOT):
    result = {}
    for r in csv.DictReader(io.StringIO((root/SAMPLING).read_text(encoding='utf-8'))):
        name = Path(r['image']).stem
        if r['sequence'] not in ('real_seq12','real_seq13','real_seq14') or name in result:
            raise ValueError('Invalid sampling map scope/duplicate')
        frame, stamp = int(r['source_frame_index']), float(r['source_time_seconds'])
        if frame < 0 or not math.isfinite(stamp) or stamp < 0 or r['sequence'] != name.rsplit('_',1)[0]:
            raise ValueError('Invalid source-frame/time metadata')
        result[name] = dict(source_video=Path(r['source_video']).name, segment=r['segment'],
                            source_frame_index=frame, source_time_seconds=stamp)
    if Counter(n.rsplit('_',1)[0] for n in result) != Counter(dict(real_seq12=141,real_seq13=304,real_seq14=149)):
        raise ValueError('Sampling map incomplete')
    return result


def annotation_audit(rows, labels, root=ROOT):
    indexed = {r['image']:r for r in labels['records']}
    if set(indexed) != {r['image'] for r in rows}:
        raise ValueError('Collection/annotation image identities differ')
    conversion = parse((root/CONVERSION).read_bytes())
    calibration = parse((root/CALIBRATION).read_bytes())
    if conversion['k'] != 2.1:
        raise ValueError('Axis conversion k differs')
    records = []
    axis_hashes = {}
    for row in rows:
        source = indexed[row['image']]
        if source['split'] != row['split'] or source['sequence'] != row['sequence']:
            raise ValueError('Collection/annotation split changed')
        path = root/DATA/relative(source['annfile'])
        data = path.read_bytes()
        if digest(data) != source['annotation_sha256']:
            raise ValueError('TRAIN/VAL annotation bytes differ: '+row['image'])
        geometry = d.polygon_geometry(data.decode('utf-8'))
        item = dict(image=row['image'], role=row['reliability_role'], sequence=row['sequence'],
            annotation_family=source['annotation_family'], annotation_sha256=digest(data),
            polygon_geometry=geometry, cached_GT_check=d.annotation_comparison(row['gt'], geometry))
        if source['annotation_family'] == 'axis_k2p1':
            meta = conversion['sequences'][row['sequence']]
            if (meta['k0'] != 2.1 or meta['model_output_used'] is not False or
                    meta['target_geometry'] != 'central_grab_structure_defined_by_axis'):
                raise ValueError('Axis conversion contract changed')
            matches = [r for r in meta['records'] if r['filename'] == row['image']+'.txt']
            if len(matches) != 1:
                raise ValueError('Axis conversion record missing/duplicate')
            record = matches[0]
            axis_path = root/DATA/'provenance/axis_k2p1'/row['sequence']/'axis_json'/relative(record['source_json'])
            axis = axis_path.read_bytes()
            if digest(axis) != record['source_json_sha256']:
                raise ValueError('Axis snapshot SHA differs')
            axis_hashes[str(axis_path.relative_to(root))] = digest(axis)
            document = parse(axis)
            shapes = [s for s in document['shapes'] if s['label'] == 'axis' and s['shape_type'] == 'line']
            if len(shapes) != 1 or len(shapes[0]['points']) != 2:
                raise ValueError('Expected one axis line')
            a,b = shapes[0]['points']
            d.finite(a,2,'axis endpoint'); d.finite(b,2,'axis endpoint')
            length = math.hypot(b[0]-a[0],b[1]-a[1])
            if length <= 0 or not math.isclose(length,record['axis_length_px'],abs_tol=1e-8,rel_tol=1e-10):
                raise ValueError('Axis length/conversion identity differs')
            gt = d.canonical(row['gt'])
            delta = [gt[0]-(a[0]+b[0])/2,gt[1]-(a[1]+b[1])/2,
                     gt[2]-length,gt[3]-length/2.1]
            item['axis_check'] = dict(axis_length_px=length, assigned_short_px=length/2.1,
                cached_GT_delta_px=delta,
                within_rounding_tolerance=all(abs(v)<=d.SETTINGS['annotation_pixel_tolerance'] for v in delta),
                short_edge_independently_annotated=False)
        records.append(item)
    groups = {}
    for seq in sorted({r['sequence'] for r in records}):
        group = [r for r in records if r['sequence'] == seq]
        groups[seq] = dict(frames=len(group), annotation_family=group[0]['annotation_family'],
            cached_GT_polygon_mismatches=sum(not r['cached_GT_check']['within_rounding_tolerance'] for r in group),
            polygon_aspect=d.distribution([r['polygon_geometry']['aspect'] for r in group]),
            axis_frames=sum('axis_check' in r for r in group),
            axis_GT_mismatches=sum(not r['axis_check']['within_rounding_tolerance'] for r in group if 'axis_check' in r))
    return dict(groups=groups, records=records, axis_snapshot_hashes=axis_hashes,
        rounding_pixel_tolerance=d.SETTINGS['annotation_pixel_tolerance'],
        annotation_policy=labels['annotation_policy'],
        frozen_depth_contract=dict(calibration_id=calibration['calibration_id'],
            calibration_sha256=sha(root/CALIBRATION),
            coordinate_contract=calibration['coordinate_contract'],
            formula=calibration['formula'], deployable_inputs=calibration['deployable_inputs'],
            formula_executed=False, parameters_refitted=False),
        physical_reference_match='UNCONFIRMED_REQUIRES_OBJECT_AND_CAMERA_GEOMETRY_EVIDENCE',
        manual_review_required=[
            'Does each port sequence axis/box refer to the same physical structure as Webots top-beam plane?',
            'Are real projected short edges independently measured or assigned by the axis-k rule?',
            'Are physical dimensions, camera intrinsics and plane/pose definitions known for real depth?',
            'Are the original unconverted real label definitions documented for each sequence?'],
        limitations=['Matching annotation bytes/axis generation does not establish physical reference equivalence.',
            'Real axis-k labels assign the short edge; they are not independent width measurements.',
            'Different sequence ratios are not proof of annotation error or a mandate for global scaling.',
            'No Webots frames, depth errors or TEST annotations used.'])


def numeric_diagnosis(rows, labels, samples, sampling):
    before = fingerprint(rows)
    labels_by_image = {r['image']:r for r in labels['records']}
    roles = {r['image']:r['role'] for r in samples['samples']}
    if len(roles) != 64 or Counter(roles.values()) != Counter(dict(fit=48,probe=16)):
        raise ValueError('Preserve original64 fit/probe roles')
    train = {r['image']:r for r in rows if r['reliability_role'] == 'train'}
    if not set(roles) <= set(train):
        raise ValueError('Legacy TRAIN sample does not belong to collection')
    for s in samples['samples']:
        if (s['sequence'] != train[s['image']]['sequence'] or s['split'] != train[s['image']]['split'] or
                s['annotation_sha256'] != labels_by_image[s['image']]['annotation_sha256']):
            raise ValueError('Legacy TRAIN sample annotation/role differs')
    records = []
    for r in rows:
        records.append(dict(image=r['image'], sequence=r['sequence'], domain=r['domain'],
            role=r['reliability_role'], frame_id=r['frame_id'],
            sampling_role=('legacy64_'+roles[r['image']] if r['image'] in roles else
                           'remaining_train' if r['reliability_role']=='train' else 'val'),
            annotation_family=labels_by_image[r['image']]['annotation_family'],
            sampling=sampling.get(r['image']),
            b=d.size_errors(r['gt'],r['b_original']), midpoint=d.size_errors(r['gt'],r['pred'])))
    groups = d.summaries(records)
    runs = d.error_runs(records)
    if fingerprint(rows) != before:
        raise ValueError('Diagnostic changed original rows')
    support = {}
    for flag in d.FLAGS:
        positive = [r for r in records if r['role']=='train' and r['midpoint'] is not None and r['midpoint'][flag]]
        support[flag] = dict(frames=len(positive), videos=dict(Counter(r['sequence'] for r in positive)),
            legacy_sampling_roles=dict(Counter(r['sampling_role'] for r in positive)),
            error_runs=sum(v['role']=='train' and v['arm']=='midpoint' and v['flag']==flag for v in runs),
            independent_scenes_confirmed=False, sufficient_support='NOT_AUTOMATICALLY_DETERMINED')
    return dict(groups=groups, train_error_support=support, records=records, error_runs=runs,
        source_rows_fingerprint=before, standardized_scale_only=True,
        proposed_loss_implemented=False, training_started=False,
        limitations=['All summaries are descriptive; no scalar cutoff, loss weight or checkpoint is selected.',
            'Overlapping error flags/runs are not independent errors or scenes.',
            'Legacy64 TRAIN probe has been used before and is not independent generalization.',
            'No automatic support sufficiency, weather attribution, depth or reliability benefit claim.'])


def run(args):
    identity, labels = checked_sources()
    out = args.out_dir.resolve()
    inputs = [ROOT/DATA, ROOT/'crane_project/tools', ROOT/'crane_project/configs']
    inputs.append(ROOT/DEFAULT_COLLECTION)
    if args.collection_dir is not None:
        inputs.append(args.collection_dir.resolve())
    if args.collection_archive is not None:
        inputs.append(args.collection_archive.resolve())
    if any(p.resolve() == out or p.resolve() in out.parents for p in inputs):
        raise ValueError('Output must be outside frozen inputs')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',protocol_document()); write(out/'source_identity.json',identity)
    try:
        if args.stage == 'check':
            write(out/'check_report.json',dict(status='SIZE_SUPPORT_STATIC_CONTRACT_PASS',
                numeric_collection_loaded=False,model_loaded=False,requires_gpu=False))
        else:
            rows, proof = load_collection(args.collection_dir,args.collection_archive)
            write(out/'input_identity.json',proof)
            samples = parse((ROOT/SAMPLES).read_bytes())
            numbers = numeric_diagnosis(rows,labels,samples,sampling_map())
            annotations = annotation_audit(rows,labels)
            write(out/'size_support.json',numbers)
            write(out/'annotation_contract.json',annotations)
            # Rehash all read inputs after analysis; don't silently accept changed labels.
            if checked_sources()[0] != identity:
                raise ValueError('Sources changed during diagnosis')
            if load_collection(args.collection_dir,args.collection_archive)[1] != proof:
                raise ValueError('Collection changed during diagnosis')
            for s in labels['records']:
                if sha(ROOT/DATA/s['annfile']) != s['annotation_sha256']:
                    raise ValueError('Annotation changed during diagnosis')
            if any(sha(ROOT/p) != expected for p,expected in annotations['axis_snapshot_hashes'].items()):
                raise ValueError('Axis snapshot changed during diagnosis')
            for role in ('train','val'):
                for domain in ('real','sim'):
                    item = numbers['groups'][role+'/domain:'+domain]['midpoint']
                    print(role,domain,'outputs',str(item['outputs'])+'/'+str(item['frames']),
                          'size_wrong',item['error_flags']['size_wrong'],
                          'short_bias',item['continuous']['short_relative']['mean'],flush=True)
            bad = sum(v['cached_GT_polygon_mismatches']+v['axis_GT_mismatches'] for v in annotations['groups'].values())
            print('annotation_geometry_mismatch_checks',bad,
                  'physical_reference_match',annotations['physical_reference_match'],flush=True)
        files = {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        status = ('SIZE_SUPPORT_STATIC_CONTRACT_PASS' if args.stage=='check' else
                  'SIZE_SUPPORT_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED')
        write(out/'completion.json',dict(protocol=d.VERSION,status=status,artifacts=files,
            scope=protocol_document()['scope'],performance_improvement_claim=False))
        print(status,str(out),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('check','diagnose'),required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--collection-dir',type=Path,help='Existing run PARENT containing collect/')
    inputs.add_argument('--collection-archive',type=Path,help='Existing analysis_20261006.tar.gz; no extraction')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
