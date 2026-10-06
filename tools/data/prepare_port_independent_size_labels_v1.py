"""TRAIN-only axis-source audit and unassigned independent corner-label packet.

This tool never writes dataset labels, assigns splits, runs models or exports GT.
The check command checks annotation structure, not physical truth or readiness.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.port_geometry_size_support_v1 import polygon_geometry

spec = importlib.util.spec_from_file_location(
    'size_reference', ROOT/'tools/data/prepare_port_size_reference_review_v1.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
VERSION = 'port_independent_size_labels_v1'
PACKET_ROOT = ROOT/'annotation_materials'
GROUPS = {'real_seq01': (339, 2.7), 'real_seq05': (560, 1.2),
          'real_seq06': (466, 1.5), 'real_seq12': (141, 2.1),
          'real_seq13': (304, 2.1)}
VIDEO_PLAN = {'1 (8).mp4': (1104, 16), '1 (11).mp4': (163, 4)}
LABEL = 'central_structure_reference'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n')


def owned(path, parent):
    result = Path(path).resolve()
    if parent.resolve() not in result.parents or result.is_symlink():
        raise ValueError('Path escapes owned directory: '+str(path))
    return result


def new_output(path):
    out = Path(path).resolve()
    if out.parent != PACKET_ROOT or not out.name.startswith(VERSION+'_'):
        raise ValueError('Use a new annotation_materials/'+VERSION+'_... directory')
    if out.exists():
        raise FileExistsError(out)
    return out


def select_frames(total, count):
    # Predeclared, evenly spaced decode indices; no predictions/error selection.
    if count < 2 or total < count:
        raise ValueError('Invalid sampling budget')
    return [round((total-1)*(.05+.90*i/(count-1))) for i in range(count)]


def axis_check(document, text, k, image_size):
    if (document.get('imageWidth'), document.get('imageHeight')) != image_size:
        raise ValueError('Axis/image dimensions differ')
    shapes = document.get('shapes')
    if not isinstance(shapes, list) or len(shapes) != 1:
        raise ValueError('Expected one source axis, no independent size annotation')
    shape = shapes[0]
    if shape.get('label') != 'axis' or shape.get('shape_type') != 'line':
        raise ValueError('Unexpected source annotation')
    points = shape.get('points')
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError('Expected two axis points')
    if not all(isinstance(p, list) and len(p) == 2 and
               all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in points):
        raise ValueError('Invalid axis coordinates')
    length = math.hypot(points[1][0]-points[0][0], points[1][1]-points[0][1])
    if length <= 0 or not math.isfinite(k) or k <= 1:
        raise ValueError('Invalid axis length/ratio')
    geom = polygon_geometry(text)
    center = [(points[0][i]+points[1][i])/2 for i in (0, 1)]
    errors = {'center_px': math.dist(center, geom['center']),
              'long_px': abs(length-geom['long_px']),
              'short_px': abs(length/k-geom['short_px'])}
    # Direction check using long polygon edge, insensitive to pi reversal.
    coords = list(map(float, text.split()[:8]))
    poly = list(zip(coords[::2], coords[1::2]))
    edges = [(poly[(i+1)%4][0]-poly[i][0], poly[(i+1)%4][1]-poly[i][1]) for i in range(4)]
    dx, dy = max(edges, key=lambda p: math.hypot(*p))
    ax, ay = points[1][0]-points[0][0], points[1][1]-points[0][1]
    errors['orientation_transverse_px'] = abs(dx*ay-dy*ax)/math.hypot(dx, dy)
    return dict(errors=errors, conversion_consistent=max(errors.values()) <= .25,
                independent_short_edge_measurement=False,
                axis_outside_image=any(not (0 <= p[0] < image_size[0] and
                                            0 <= p[1] < image_size[1]) for p in points))


def audit_train(dataset, inputs):
    from PIL import Image
    legacy_path = dataset/'provenance/axis_legacy_train_v1/manifest.json'
    new_path = dataset/'provenance/axis_k2p1/conversion_manifest.json'
    dataset_path = dataset/'manifest.json'
    allow_path = ROOT/'crane_project/tools/port_geometry_size_support_v1_labels.json'
    for path in (legacy_path, new_path, dataset_path, allow_path):
        inputs[str(path)] = sha(path)
    old = json.loads(legacy_path.read_text())
    new = json.loads(new_path.read_text())
    records = {r['id']: r for r in json.loads(dataset_path.read_text())['records']
               if r['split'] == 'train' and r['sequence'] in GROUPS}
    if old['evidence_role'] != 'recovered_native_train_axis_labels_not_independent_gt':
        raise ValueError('Legacy evidence role differs')
    allowed = {Path(r['annfile']).stem: r for r in json.loads(allow_path.read_text())['records']
               if r['split'] == 'train' and r['sequence'] in GROUPS}
    expected = {p.stem for group in GROUPS
                for p in (dataset/'train/annfiles').glob(group+'_*.txt')}
    if set(records) != expected or set(allowed) != expected or len(expected) != 1810:
        raise ValueError('Frozen real TRAIN identity/count differs')
    rows = []; groups = {}
    for group, (count, k) in GROUPS.items():
        ids = sorted(stem for stem in expected if stem.startswith(group+'_'))
        if len(ids) != count:
            raise ValueError('TRAIN group count differs')
        new_rows = {} if group in old['sequences'] else {
            Path(r['filename']).stem: r for r in new['sequences'][group]['records']}
        if new_rows:
            meta = new['sequences'][group]
            if (meta['k0'] != k or meta['model_output_used'] is not False or
                    meta['target_geometry'] != 'central_grab_structure_defined_by_axis' or
                    set(new_rows) != set(ids)):
                raise ValueError('New-axis provenance differs')
        for stem in ids:
            ann = dataset/'train/annfiles'/(stem+'.txt')
            image = dataset/'train/images'/(stem+'.jpg')
            rec = records[stem]
            if rec['images']['path'] != str(image.relative_to(dataset)) or rec['annfiles']['path'] != str(ann.relative_to(dataset)):
                raise ValueError('TRAIN manifest path differs')
            if new_rows:
                source = new_rows[stem]
                axis = dataset/'provenance/axis_k2p1'/group/'axis_json'/source['source_json']
                axis_sha = source['source_json_sha256']
                binding = 'SNAPSHOT_AND_CURRENT_TRAIN_IMAGE_BOUND'
            else:
                source = old['rows'][stem]
                if source['split'] != 'train' or source['sequence'] != group or source['conversion_k'] != k:
                    raise ValueError('Legacy TRAIN provenance differs')
                axis = dataset/'provenance/axis_legacy_train_v1'/group/'axis_json'/(stem+'.json')
                axis_sha = source['axis_json_sha256']
                binding = source['original_source_image_equality']
                if source['image_sha256'] != rec['images']['sha256'] or source['annotation_sha256'] != rec['annfiles']['sha256']:
                    raise ValueError('Legacy manifest/image binding differs')
            axis = owned(axis, dataset/'provenance')
            for path, digest in ((image, rec['images']['sha256']), (ann, rec['annfiles']['sha256']), (axis, axis_sha)):
                if sha(path) != digest:
                    raise ValueError('Input SHA differs: '+str(path))
                inputs[str(path)] = digest
            if allowed[stem]['annotation_sha256'] != rec['annfiles']['sha256']:
                raise ValueError('Fixed annotation allowlist differs')
            document = json.loads(axis.read_text())
            if Path(document.get('imagePath', '')).stem not in (stem, stem.replace('real_', '', 1)):
                raise ValueError('Axis imagePath binding differs')
            with Image.open(image) as im:
                result = axis_check(document, ann.read_text(), k, im.size)
            rows.append(dict(image=stem, sequence=group, role='EXISTING_TRAIN_SOURCE_AUDIT',
                k=k, axis_json=str(axis.relative_to(dataset)), source_image_equality=binding, **result))
        subset = [r for r in rows if r['sequence'] == group]
        groups[group] = dict(count=count, k=k,
            conversion_mismatches=sum(not r['conversion_consistent'] for r in subset),
            axis_outside_image=sum(r['axis_outside_image'] for r in subset),
            max_errors_px={key: max(r['errors'][key] for r in subset) for key in subset[0]['errors']})
    return dict(protocol=VERSION, groups=groups, rows=rows,
        total_real_train=len(rows), axis_only_sources=len(rows),
        independently_measured_short_edges=0,
        physical_reference_match='UNCONFIRMED',
        scope='snapshot_generation_consistency_not_independent_label_accuracy')


def validate_annotation(document, frame):
    """Return REVIEWED/PENDING/UNMEASURABLE; invalid labels raise ValueError.

    Manual declarations cannot prove accurate physical/corner identification.
    No fixed-aspect test, rectangularization or inferred hidden corners.
    """
    if (document.get('imagePath') != Path(frame['image']).name or
        (document.get('imageWidth'), document.get('imageHeight')) != tuple(frame['size'])):
        raise ValueError('Annotation source image identity differs')
    flags = document.get('flags', {})
    reviewed = flags.get('independent_reviewed', False)
    unmeasurable = flags.get('unmeasurable', False)
    if type(reviewed) is not bool or type(unmeasurable) is not bool:
        raise ValueError('Review flags must be Boolean')
    shapes = document.get('shapes')
    if not isinstance(shapes, list):
        raise ValueError('Shapes must be a list')
    if unmeasurable:
        if shapes or not reviewed or not document.get('review_note', '').strip():
            raise ValueError('Unmeasurable requires reviewed, no guessed corners, and a reason')
        return dict(status='UNMEASURABLE', independent_corner_observation=False)
    if not shapes:
        if reviewed:
            raise ValueError('Reviewed measurable image needs four visible corners')
        return dict(status='PENDING', independent_corner_observation=False)
    if len(shapes) != 1:
        raise ValueError('Expected exactly one independent quadrilateral')
    shape = shapes[0]
    if shape.get('shape_type') != 'polygon' or shape.get('label') != LABEL:
        raise ValueError('Axis/rectangle/old detector labels are not independent corner labels')
    points = shape.get('points')
    if not isinstance(points, list) or len(points) != 4 or not all(
        isinstance(p, list) and len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in p)
        for p in points):
        raise ValueError('Exactly four finite original-image coordinates required')
    width, height = frame['size']
    if not all(0 <= p[0] < width and 0 <= p[1] < height for p in points):
        raise ValueError('Corner outside original image')
    cross = []
    for i in range(4):
        a, b, c = points[i], points[(i+1)%4], points[(i+2)%4]
        cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    if not (all(v > 1e-6 for v in cross) or all(v < -1e-6 for v in cross)):
        raise ValueError('Corners must be cyclic, convex, nondegenerate; no crossed edges')
    if not reviewed:
        return dict(status='PENDING_REVIEW', independent_corner_observation=False)
    if (not document.get('object_definition', '').strip() or
            document.get('measurement_basis') != 'manual_visible_four_corners'):
        raise ValueError('Independent review needs object definition and visible-corner declaration')
    edges = [math.dist(points[i], points[(i+1)%4]) for i in range(4)]
    sides = sorted(((edges[0]+edges[2])/2, (edges[1]+edges[3])/2), reverse=True)
    return dict(status='REVIEWED_CORNER_OBSERVATION', independent_corner_observation=True,
        visible_opposite_edge_means_px=sides,
        scope='projected_quadrilateral_edges_not_native_OBB_or_physical_dimensions')


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def prepare(video_root, out):
    from PIL import Image, ImageDraw, ImageOps
    dataset = ROOT/'crane_project/data/crane_grab_port_day2night_v1'
    inputs = {str(Path(__file__)): sha(__file__),
        str(ROOT/'tests/test_port_independent_size_labels_v1.py'): sha(ROOT/'tests/test_port_independent_size_labels_v1.py'),
        str(ROOT/'tools/data/prepare_port_size_reference_review_v1.py'): sha(ROOT/'tools/data/prepare_port_size_reference_review_v1.py'),
        str(ROOT/'crane_project/utils/port_geometry_size_support_v1.py'): sha(ROOT/'crane_project/utils/port_geometry_size_support_v1.py'),
        str(ROOT/'crane_project/utils/port_midpoint_depth_v1.py'): sha(ROOT/'crane_project/utils/port_midpoint_depth_v1.py')}
    audit = audit_train(dataset, inputs)
    sampling = dataset/'provenance/axis_k2p1/sampling_manifest.json'
    inputs[str(sampling)] = sha(sampling)
    used = {Path(k).name: v for k, v in json.loads(sampling.read_text())['source_sha256'].items()}
    videos = []
    for name, (expected_count, budget) in VIDEO_PLAN.items():
        path = video_root/name; digest = sha(path)
        reference.validate_video(name, digest, used); inputs[str(path)] = digest
        meta = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', str(path)]))['streams'][0]
        if int(meta['nb_read_frames']) != expected_count or (meta['width'], meta['height']) != (1920, 1088):
            raise ValueError('Decoded candidate metadata differs')
        videos.append(dict(name=name, source=str(path), sha256=digest,
            size=[meta['width'], meta['height']], decoded_frame_count=expected_count,
            frame_indices=select_frames(expected_count, budget),
            role='UNASSIGNED_ANNOTATION_CANDIDATE', event_independence='UNCONFIRMED',
            time_basis='decode_index_not_wall_clock_or_physical_sampling_rate'))
    out.parent.mkdir(exist_ok=True)
    out.mkdir(); (out/'images').mkdir(); (out/'annotations').mkdir()
    write_json(out/'train_axis_audit.json', audit)
    frames = []
    for video in videos:
        prefix = 'candidate'+('08' if video['name'] == '1 (8).mp4' else '11')
        temp = out/(prefix+'_decode'); temp.mkdir()
        selector = '+'.join('eq(n\\,'+str(i)+')' for i in video['frame_indices'])
        run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', video['source'],
             '-vf', 'select='+selector, '-vsync', '0', '-frames:v', str(len(video['frame_indices'])),
             str(temp/'%05d.png')])
        decoded = sorted(temp.glob('*.png'))
        if len(decoded) != len(video['frame_indices']):
            raise ValueError('Missing decoded candidate frame')
        columns = 4 if len(decoded) > 4 else 2
        page = Image.new('RGB', (columns*400, math.ceil(len(decoded)/columns)*255+30), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((8, 8), prefix+' UNASSIGNED; no labels/predictions; original full-resolution PNG in images/', fill='black')
        for j, (source, index) in enumerate(zip(decoded, video['frame_indices'])):
            stem = prefix+'_f'+str(index).zfill(5)
            image = out/'images'/(stem+'.png'); source.rename(image)
            with Image.open(image) as im:
                if list(im.size) != video['size']:
                    raise ValueError('Decoded image dimensions differ')
                page.paste(ImageOps.contain(im.convert('RGB'), (390, 221)), ((j%columns)*400+5, (j//columns)*255+32))
            draw.text(((j%columns)*400+5, (j//columns)*255+256), stem, fill='black')
            ann = out/'annotations'/(stem+'.json')
            write_json(ann, dict(version='3.3.10', flags=dict(independent_reviewed=False, unmeasurable=False),
                shapes=[], imagePath='../images/'+image.name, imageData=None,
                imageWidth=video['size'][0], imageHeight=video['size'][1],
                object_definition='', measurement_basis='', review_note=''))
            frames.append(dict(image=str(image.relative_to(out)), image_sha256=sha(image), size=video['size'],
                annotation=str(ann.relative_to(out)), source_video=video['name'],
                source_video_sha256=video['sha256'], decode_index=index, role=video['role']))
        temp.rmdir()  # Our own now-empty decode directory only.
        page.save(out/(prefix+'_overview.png'))
    # Templates live next to images; path is relative to annotations/.
    receipt = dict(protocol=VERSION, videos=videos, frames=frames, inputs_sha256=inputs,
        annotation_label=LABEL, ready_for_training=False, split_assigned=False,
        immutable_train_labels=True, physical_reference_match='UNCONFIRMED',
        tool_limits='No GT export/model/error/depth evaluation; manual labels and event review still required')
    write_json(out/'packet.json', receipt)
    (out/'ANNOTATION_GUIDE.md').write_text(
        '# 独立四角点标注材料：尚未进入TRAIN\n\n'
        'images/是20张原分辨率PNG；annotations/是空白JSON，可在标注器中指定该输出目录。'
        '同一物理对象、同一参考面须先统一定义；不要直接默认外侧抓斗轮廓或固定比例框就是深度参考面。\n\n'
        '仅在四角点都能独立识别时，以polygon类型、central_structure_reference标签，按顺／逆时针依次标4点。'
        '坐标为原图像素。禁止用预测框、旧固定比例、隐蔽角点推算或自动传播结果代替独立人工复核。\n\n'
        '逐张填写object_definition（具体部件及四个角点身份）、measurement_basis='
        'manual_visible_four_corners；确认后flags.independent_reviewed=true。'
        '遮挡或模糊不能确定时保留shapes=[]，置unmeasurable=true、independent_reviewed=true，'
        '并在review_note写原因。未处理的JSON保持PENDING。\n\n'
        '标注器可能丢弃自定义字段，保存后检查这些字段。保留packet.json、artifacts.json及原PNG不变；'
        '仅annotations/JSON允许人工修改。check只检查结构、身份及声明，不验证人工标注真值。\n\n'
        '这些是无split的候选观测，不导出DOTA、不覆盖旧GT、不启动训练。'
        '角点观测的对边均长也不等同于原生OBB长短边或米制尺寸；后续仍需对象、参考面、'
        '标注复核、近重复／事件级角色及模型错误支持检查。容器帧编号不保证物理时间间隔。\n')
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Source changed during preparation: '+path)
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'artifacts.json', files)
    print('INDEPENDENT_SIZE_PACKET_PREPARED', len(frames), 'real TRAIN axis-only', audit['axis_only_sources'],
          'conversion mismatches', sum(g['conversion_mismatches'] for g in audit['groups'].values()))


def check(packet_dir):
    packet_dir = Path(packet_dir).resolve()
    if packet_dir.parent != PACKET_ROOT or not packet_dir.name.startswith(VERSION+'_'):
        raise ValueError('Only owned annotation packet under annotation_materials is allowed')
    if not (packet_dir/'packet.json').is_file():
        raise FileNotFoundError('Missing annotation packet: '+str(packet_dir/'packet.json')+
            '. Commit/push the complete annotation_materials packet on the source machine, '
            'then git pull on the server; check does not generate images or labels.')
    receipt = json.loads((packet_dir/'packet.json').read_text())
    artifacts = json.loads((packet_dir/'artifacts.json').read_text())
    if sha(packet_dir/'packet.json') != artifacts['packet.json'] or receipt['protocol'] != VERSION:
        raise ValueError('Packet identity changed')
    if receipt['ready_for_training'] is not False or receipt['split_assigned'] is not False:
        raise ValueError('Packet cannot promote labels or assign split')
    expected = {owned(packet_dir/f['annotation'], packet_dir/'annotations') for f in receipt['frames']}
    if {p.resolve() for p in (packet_dir/'annotations').glob('*.json')} != expected:
        raise ValueError('Unexpected/missing annotation JSON outside frozen frame plan')
    rows = []
    for frame in receipt['frames']:
        image = owned(packet_dir/frame['image'], packet_dir/'images')
        annotation = owned(packet_dir/frame['annotation'], packet_dir/'annotations')
        if sha(image) != frame['image_sha256']:
            raise ValueError('Annotation image pixels changed')
        document = json.loads(annotation.read_text())
        # Accept an editor's basename or the original safe relative imagePath.
        if document.get('imagePath') not in (image.name, '../images/'+image.name):
            raise ValueError('Annotation points to wrong source image')
        document['imagePath'] = image.name
        rows.append(dict(image=frame['image'], annotation_sha256=sha(annotation),
                         **validate_annotation(document, frame)))
    return dict(protocol=VERSION, rows=rows,
        status_counts={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
        ready_for_training=False, split_assigned=False,
        limitations='Structural validation only; corner accuracy, common object, event independence and downstream benefit unverified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare'); p.add_argument('--video-root', type=Path, required=True)
    p.add_argument('--out-dir', required=True)
    p = subs.add_parser('check'); p.add_argument('--packet-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.video_root, new_output(args.out_dir))
        else:
            print(json.dumps(check(args.packet_dir), ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, 'ERROR: '+str(error)+'\n')


if __name__ == '__main__':
    main()
