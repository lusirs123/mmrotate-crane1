"""Check pinned official sources and prepare YOLO labels. Never train/infer.

Run from any directory. Default action checks configuration/source identity.
YOLO label conversion uses only TRAIN and VAL unless --include-test is explicit.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.check_faa_adaptation_v1 import resolve

DATA = ROOT / 'crane_project/data/crane_grab_port_day2night_v1'
MANIFEST_SHA = '128bd9aac390ad878a46c005a22f44d322a5acf61a47ad128991eb7444a4886f'
EXPECTED = dict(train=1810, train_sim=748, val=887, test=1440)
GAUCHO_CONFIG = 'configs/gaucho/crane_gaucho_fcos_r50_probiou_port_day2night_v1.py'
YOLO_CONFIG = 'configs/yolo26/crane_yolo26m_obb_port_day2night_v1.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def dataset_manifest():
    if sha(DATA / 'manifest.json') != MANIFEST_SHA:
        raise ValueError('Frozen dataset manifest changed')
    manifest = read(DATA / 'manifest.json')
    rows = manifest['records']
    if dict(Counter(r['split'] for r in rows)) != EXPECTED:
        raise ValueError('Frozen dataset counts differ')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate frame identity')
    splits = {}
    for row in rows:
        splits.setdefault(row['images']['sha256'], set()).add(row['split'])
    if any(len(value) > 1 for value in splits.values()):
        raise ValueError('Cross-split image duplicate in manifest')
    return rows


def check_sources():
    report = {}
    for name in ('gaucho', 'yolo26'):
        spec = read(ROOT / 'configs' / name / 'upstream_manifest.json')
        for path, item in spec['files'].items():
            if sha(ROOT / path) != item['sha256']:
                raise ValueError('Official source drift: ' + path)
            if item.get('adaptations'):
                restored = (ROOT / path).read_text()
                for change in reversed(item['adaptations']):
                    if restored.count(change['local']) != 1:
                        raise ValueError('Unexpected official import/registry adaptation: ' + path)
                    restored = restored.replace(change['local'], change['upstream'])
                if hashlib.sha256(restored.encode()).hexdigest() != item['upstream_sha256']:
                    raise ValueError('GauCho differs from official source beyond declared aliases: ' + path)
        for path, item in spec.get('shared_dependencies', {}).items():
            if sha(ROOT / path) != item['local_sha256']:
                raise ValueError('Shared GauCho dependency drift: ' + path)
        report[name] = dict(repository=spec['repository'], commit=spec['commit'],
                            official_files_verified=len(spec['files']),
                            shared_dependencies_verified=len(spec.get('shared_dependencies', {})))
    return report


def check_configs():
    cfg, closure = resolve(ROOT / GAUCHO_CONFIG)
    native, _ = resolve(ROOT / 'configs/gaucho/gaucho_anchorless_dotav1/'
                         'gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py')
    expected_model = deepcopy(native['model'])
    expected_model['bbox_head']['num_classes'] = 1
    expected_model['bbox_head']['loss_bbox']['type'] = 'GauChoGDLossV1'
    if cfg['model'] != expected_model:
        raise ValueError('GauCho model changed beyond class count and registry-only alias')
    if cfg['optimizer'] != native['optimizer']:
        raise ValueError('Official GauCho optimizer changed')
    faa, _ = resolve(ROOT / 'configs/faa/crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_v1.py')
    if cfg['data'] != faa['data']:
        raise ValueError('GauCho/FAA dataset or preprocessing differs')
    if cfg['evaluation']['save_best'] is not None:
        raise ValueError('Online VAL must not select formal weights')
    if cfg['runner']['max_epochs'] != 24 or cfg['lr_config']['step'] != [16, 22]:
        raise ValueError('GauCho preparation budget differs')
    yolo = read(ROOT / YOLO_CONFIG)
    original = (ROOT / 'configs/yolo26/official/yolo26-obb.yaml').read_text()
    expected_yaml = original.replace('nc: 80 # number of classes',
                                     'nc: 1 # single grab class; official architecture unchanged') + '\nscale: m\n'
    if (ROOT / yolo['architecture']).read_text() != expected_yaml:
        raise ValueError('YOLO26 architecture changed beyond nc/scale')
    if yolo['initialization']['allow_detection_pretraining']:
        raise ValueError('This comparison uses ImageNet-only initialization')
    if yolo['selection']['auto_test'] or yolo['train_args']['resume']:
        raise ValueError('Automatic TEST/resume is not part of preparation')
    candidates = yolo['selection']['candidate_completed_epochs']
    if yolo['selection']['native_checkpoint_names'] != ['epoch%d.pt' % (e-1) for e in candidates]:
        raise ValueError('YOLO periodic checkpoints are zero-based')
    data_yaml = (ROOT / yolo['data']).read_text()
    if '\ntest:' in data_yaml or '\ndownload:' in data_yaml or '\npath:' in data_yaml:
        raise ValueError('Training YAML must be portable and TRAIN/VAL-only')
    return dict(gaucho_config=GAUCHO_CONFIG, gaucho_inheritance=closure,
                gaucho_model_delta=['bbox_head.num_classes: 15 -> 1'],
                gaucho_registry_aliases={'GDLoss_v1': 'GauChoGDLossV1'},
                yolo_config=YOLO_CONFIG, yolo_model_delta=['nc: 80 -> 1', 'scale: m'],
                initialization_sha_recorded=yolo['initialization']['sha256'] is not None,
                training_authorized=False, runtime_validated=False)


def polygon_label(text, width, height):
    """DOTA polygon -> official YOLO OBB corners, without clipping/reboxing."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) not in (9, 10):
        raise ValueError('Expected one single-class DOTA annotation')
    fields = lines[0]
    # The existing CraneDataset includes both difficulty 0 and 1. Preserve all
    # those objects, rather than silently treating flag 1 as an ignored GT.
    if fields[8] != 'grab' or (len(fields) == 10 and fields[9] not in ('0', '1')):
        raise ValueError('Unexpected class or difficulty flag')
    values = [float(value) for value in fields[:8]]
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid dimensions or nonfinite polygon')
    xy = list(zip(values[::2], values[1::2]))
    if any(not (0 <= x <= width and 0 <= y <= height) for x, y in xy):
        raise ValueError('Out-of-image polygon must not be silently clipped')
    cross = []
    for i in range(4):
        a, b, c = xy[i], xy[(i+1) % 4], xy[(i+2) % 4]
        cross.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
    if not (all(v > 0 for v in cross) or all(v < 0 for v in cross)):
        raise ValueError('Degenerate or non-cyclic polygon')
    norm = [v / (width if i % 2 == 0 else height) for i, v in enumerate(values)]
    return '0 ' + ' '.join(format(v, '.17g') for v in norm) + '\n'


def convert_yolo(include_test=False, copy_images=False):
    from PIL import Image
    spec = read(ROOT / YOLO_CONFIG)
    destination = ROOT / spec['converted_data']
    rows = [r for r in dataset_manifest() if include_test or r['split'] != 'test']
    if destination.exists():
        raise FileExistsError('Converted dataset already exists. Check it rather than overwrite: ' + str(destination))
    staging = Path(tempfile.mkdtemp(prefix='.yolo26_prepare_', dir=str(destination.parent)))
    records = []
    try:
        for row in sorted(rows, key=lambda r: (r['split'], r['id'])):
            split = 'train' if row['split'] in ('train', 'train_sim') else row['split']
            image = DATA / row['images']['path']
            ann = DATA / row['annfiles']['path']
            if sha(image) != row['images']['sha256'] or sha(ann) != row['annfiles']['sha256']:
                raise ValueError('Dataset bytes differ from sealed manifest: ' + row['id'])
            with Image.open(image) as im:
                width, height = im.size
            label = polygon_label(ann.read_text(), width, height)
            target = staging / split / 'images' / (row['id'] + '.jpg')
            target.parent.mkdir(parents=True, exist_ok=True)
            label_path = staging / split / 'labels' / (row['id'] + '.txt')
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copyfile(image, target)
            else:
                # Staging/final directories have equal depth. Links stay relative
                # to the project, so an identical tree works on Mac and server.
                target.symlink_to(os.path.relpath(str(image), str(target.parent)))
            label_path.write_text(label)
            records.append(dict(id=row['id'], source_split=row['split'], split=split,
                                width=width, height=height,
                                image=str(target.relative_to(staging)),
                                label=str(label_path.relative_to(staging)),
                                label_sha256=sha(label_path),
                                image_sha256=row['images']['sha256'],
                                annotation_sha256=row['annfiles']['sha256']))
        write_new(staging / 'conversion_receipt.json', dict(
            protocol='crane_official_yolo_obb_conversion_v1', manifest_sha256=MANIFEST_SHA,
            image_mode='copy' if copy_images else 'relative_symlink',
            test_included=include_test, training_updates=0,
            counts=dict(Counter(r['split'] for r in records)), records=records))
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return check_converted()


def check_converted():
    from PIL import Image
    destination = ROOT / read(ROOT / YOLO_CONFIG)['converted_data']
    receipt = read(destination / 'conversion_receipt.json')
    if receipt['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Converted dataset identity differs')
    source = {r['id']: r for r in dataset_manifest()
              if receipt['test_included'] or r['split'] != 'test'}
    records = receipt['records']
    maximum_roundtrip_error = 0.0
    if len(records) != len(source) or {r['id'] for r in records} != set(source):
        raise ValueError('Converted frame set differs')
    for split in ('train', 'val', 'test'):
        expected = {r['id'] for r in records if r['split'] == split}
        for kind, suffix in (('images', '.jpg'), ('labels', '.txt')):
            observed = {p.stem for p in (destination / split / kind).glob('*' + suffix)}
            if observed != expected:
                raise ValueError('Unexpected/missing converted files: ' + split + '/' + kind)
    for row in records:
        original = source[row['id']]
        split = 'train' if original['split'] in ('train', 'train_sim') else original['split']
        if row['source_split'] != original['split'] or row['split'] != split:
            raise ValueError('Converted split substitution')
        image, label = destination / row['image'], destination / row['label']
        if image != destination / split / 'images' / (row['id'] + '.jpg') or label != destination / split / 'labels' / (row['id'] + '.txt'):
            raise ValueError('Converted path substitution')
        if sha(image) != original['images']['sha256'] or sha(DATA / original['annfiles']['path']) != original['annfiles']['sha256']:
            raise ValueError('Source image/annotation changed')
        with Image.open(image) as im:
            width, height = im.size
        expected = polygon_label((DATA / original['annfiles']['path']).read_text(), width, height)
        if label.read_text() != expected or sha(label) != row['label_sha256']:
            raise ValueError('Converted annotation differs: ' + row['id'])
        original_xy = [float(v) for v in (DATA / original['annfiles']['path']).read_text().split()[:8]]
        normalized_xy = [float(v) for v in label.read_text().split()[1:]]
        error = max(abs(a - b * (width if i % 2 == 0 else height))
                    for i, (a, b) in enumerate(zip(original_xy, normalized_xy)))
        maximum_roundtrip_error = max(maximum_roundtrip_error, error)
        if error > 1e-9:
            raise ValueError('Coordinate normalization roundtrip differs')
    counts = dict(Counter(r['split'] for r in records))
    if counts != receipt['counts']:
        raise ValueError('Converted count receipt differs')
    return dict(counts=counts, pixels_unchanged=True, coordinates_roundtrip_verified=True,
                maximum_coordinate_roundtrip_error_px=maximum_roundtrip_error,
                test_included=receipt['test_included'], receipt_sha256=sha(destination / 'conversion_receipt.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('check', 'convert-yolo', 'check-yolo'), default='check')
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--copy-images', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = dict(sources=check_sources(), configs=check_configs(),
                  manifest_sha256=MANIFEST_SHA, counts=dict(Counter(r['split'] for r in dataset_manifest())),
                  training_updates=0, inference_runs=0)
    files = [str(p.relative_to(ROOT)) for folder in ('configs/gaucho', 'configs/yolo26')
             for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.json', '.yaml')]
    files += ['crane_project/tools/' + name + '.py' for name in (
        'prepare_external_obb_v1', 'prepare_yolo26_initialization_v1',
        'check_gaucho_runtime_v1', 'check_yolo26_runtime_v1',
        'export_external_obb_v1', 'evaluate_external_obb_saved_v1', 'select_external_obb_val_v1')]
    files += ['crane_project/utils/' + name + '.py' for name in (
        'external_obb_io_v1', 'yolo26_imagenet_v1')]
    files += ['mmrotate/models/dense_heads/gaussian_fcos_head.py',
              'mmrotate/models/dense_heads/gaucho_rotated_fcos_head.py',
              'mmrotate/models/losses/gaucho_gaussian_dist_loss_v1.py']
    files += ['tests/test_external_obb_adaptation_v1.py']
    result['adaptation_sha256'] = {name: sha(ROOT / name) for name in sorted(files)}
    if args.action == 'convert-yolo':
        result['converted'] = convert_yolo(args.include_test, args.copy_images)
    elif args.action == 'check-yolo':
        result['converted'] = check_converted()
    elif args.include_test or args.copy_images:
        parser.error('Conversion flags require --action convert-yolo')
    if args.out:
        write_new(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
