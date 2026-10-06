"""Read-only physical-reference audit and TRAIN annotation-review material.

No inference, labels, split assignment, depth fitting or training. Candidate
videos8/11 remain unassigned. Review images are copies under work_dirs only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_midpoint_depth_v1 as depth

VERSION = 'port_size_reference_review_v1'
WORLD_SHA = '96ca6422cfa9625051b9895130b97fb074e68d88d76fd1b892f60c41b69f8750'
TRAIN_GROUPS = ('real_seq01', 'real_seq05', 'real_seq06', 'real_seq12',
                'real_seq13', 'sim_seq08')
CANDIDATES = ('1 (8).mp4', '1 (11).mp4')
CANDIDATE_SHA = {
    '1 (8).mp4': '3bb41956bf50ac7b4b27502f31555fa0e0631fe0a9be141a606e71555574aad5',
    '1 (11).mp4': '0f1f156a52a21b3b917bb36d0a777e10d0c794c458dd5d9fac8279db698f133c',
}


def validate_video(name, digest, used_by_name):
    if name in CANDIDATES:
        valid = digest == CANDIDATE_SHA[name] and digest not in used_by_name.values()
    else:
        valid = name == '1 (7).mp4' and digest == used_by_name.get(name)
    if not valid:
        raise ValueError('Candidate/used video identity unexpected: '+name)


def vector(text, name):
    # Pinned world: these four points share one local coordinate system.
    found = re.findall(r'DEF\s+'+re.escape(name)+r'\s+Pose\s*\{\s*translation\s+'
                       r'([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)', text)
    if len(found) != 1:
        raise ValueError('Missing or ambiguous reference node: '+name)
    result = tuple(map(float, found[0]))
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Nonfinite geometry')
    return result


def geometry(text):
    points = {n: vector(text, 'GRAB_PT_'+n) for n in ('TL', 'TR', 'BL', 'BR')}
    distance = lambda a, b: math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    long_m = (distance(points['TL'], points['TR'])+distance(points['BL'], points['BR']))/2
    short_m = (distance(points['TL'], points['BL'])+distance(points['TR'], points['BR']))/2
    center = tuple(sum(p[i] for p in points.values())/4 for i in range(3))
    if not 0 < short_m < long_m:
        raise ValueError('Invalid long-axis identity')
    return dict(corners_local_m=points, long_edge_mean_m=long_m,
                short_edge_mean_m=short_m, reference_ar_3d=long_m/short_m,
                reference_center_error_m=distance(center, vector(text, 'GRAB_REFERENCE')))


def reference(world, manifest):
    if depth.sha(world) != WORLD_SHA or depth.sha(manifest) != depth.MANIFEST_SHA:
        raise ValueError('Frozen world/depth manifest identity differs')
    result = geometry(Path(world).read_text())
    document = json.loads(Path(manifest).read_text())
    if document['sequence_id'] != depth.SEQUENCE or document['split'] != 'calibration_train':
        raise ValueError('Wrong depth sequence role')
    for key in ('long_edge_mean_m', 'short_edge_mean_m', 'reference_ar_3d', 'reference_center_error_m'):
        if not math.isclose(result[key], document['obb_reference_geometry'][
                'grab_reference_center_error_m' if key == 'reference_center_error_m' else key],
                rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError('World and depth manifest geometry differ: '+key)
    # Pinned camera block; no camera pose is needed for physical edge lengths.
    camera = re.search(r'DEF CRANE_CAMERA Camera\s*\{([\s\S]*?)recognition Recognition',
                       Path(world).read_text()).group(1)
    values = {k: float(re.search(r'\b'+k+r'\s+([-+\deE.]+)', camera).group(1))
              for k in ('width', 'height', 'fieldOfView')}
    f = values['width']/(2*math.tan(values['fieldOfView']/2))
    intrinsics = document['camera']['intrinsics']
    if values['width'] != document['camera']['width_px'] or values['height'] != document['camera']['height_px']:
        raise ValueError('Camera resolution differs')
    if not all(math.isclose(f, intrinsics[k], abs_tol=1e-9) for k in ('fx', 'fy')):
        raise ValueError('Camera focal length differs')
    result.update(focal_px=f, status='FROZEN_WEBOTS_REFERENCE_SOURCE_MATCH',
                  detection_sim08_same_capture='UNCONFIRMED',
                  real_structure_physical_match='UNCONFIRMED')
    return result


def train_samples(dataset):
    selected = []
    for group in TRAIN_GROUPS:
        split = 'train_sim' if group.startswith('sim_') else 'train'
        paths = sorted((dataset/split/'annfiles').glob(group+'_*.txt'))
        if len(paths) < 4:
            raise ValueError('Missing TRAIN group '+group)
        for i in (0, len(paths)//3, 2*len(paths)//3, len(paths)-1):
            ann = paths[i]
            image = dataset/split/'images'/(ann.stem+'.jpg')
            if not image.is_file():
                raise ValueError('Missing TRAIN image '+str(image))
            selected.append((group, image, ann))
    return selected


def output_path(path):
    out = Path(path).resolve()
    if out.parent != ROOT/'work_dirs' or not out.name.startswith(VERSION+'_'):
        raise ValueError('Output must be a new work_dirs/'+VERSION+'_... directory')
    if out.exists():
        raise FileExistsError(out)
    return out


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-root', required=True, type=Path)
    p.add_argument('--world', required=True, type=Path)
    p.add_argument('--depth-manifest', required=True, type=Path)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    out = output_path(args.out_dir)
    physical = reference(args.world, args.depth_manifest)
    dataset = ROOT/'crane_project/data/crane_grab_port_day2night_v1'
    sampling_path = dataset/'provenance/axis_k2p1/sampling_manifest.json'
    conversion_path = dataset/'provenance/axis_k2p1/conversion_manifest.json'
    sampling = json.loads(sampling_path.read_text())
    used_by_name = {Path(name).name: digest for name, digest in sampling['source_sha256'].items()}
    selected = train_samples(dataset)
    inputs = {str(f): depth.sha(f) for f in (args.world, args.depth_manifest,
              sampling_path, conversion_path, Path(__file__), ROOT/'crane_project/utils/port_midpoint_depth_v1.py',
              ROOT/'tests/test_port_size_reference_review_v1.py')}
    for _, image, ann in selected:
        inputs[str(image)] = depth.sha(image); inputs[str(ann)] = depth.sha(ann)
    video_info = []
    for name in CANDIDATES+('1 (7).mp4',):
        path = args.video_root/name
        digest = depth.sha(path)
        validate_video(name, digest, used_by_name)
        inputs[str(path)] = digest
        meta = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:format=duration', '-of', 'json', str(path)]))
        # Some source containers omit nb_frames. Count decodable frames rather
        # than inventing duration*FPS or silently dropping the video.
        decoded = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)]))
        count = int(decoded['streams'][0]['nb_read_frames'])
        if count < 4:
            raise ValueError('Too few decodable frames: '+name)
        video_info.append(dict(name=name, source=str(path), sha256=digest,
            role='UNASSIGNED_REVIEW_CANDIDATE' if name in CANDIDATES else 'EXISTING_TRAIN_REFERENCE',
            stream=meta['streams'][0], duration_seconds=float(meta['format']['duration']),
            decoded_frame_count=count,
            recording_event_independence='UNCONFIRMED'))
    from PIL import Image, ImageDraw, ImageOps
    out.mkdir(); (out/'review').mkdir()
    review = []; templates = []
    for group in TRAIN_GROUPS:
        page = Image.new('RGB', (960, 1060), 'white'); drawing = ImageDraw.Draw(page)
        drawing.text((12, 8), group+' TRAIN reference review: unmarked LEFT / existing label RIGHT', fill='black')
        for row, (_, image, ann) in enumerate(x for x in selected if x[0] == group):
            with Image.open(image) as raw:
                raw = raw.convert('RGB')
                fields = ann.read_text().split()
                if len(fields) != 10 or fields[8] != 'grab':
                    raise ValueError('Unexpected single OBB annotation')
                poly = [(float(fields[i]), float(fields[i+1])) for i in range(0, 8, 2)]
                xs, ys = zip(*poly); cx, cy = sum(xs)/4, sum(ys)/4
                side = max(220, 1.8*max(max(xs)-min(xs), max(ys)-min(ys)))
                bounds = (max(0, math.floor(cx-side/2)), max(0, math.floor(cy-side/2)),
                          min(raw.width, math.ceil(cx+side/2)), min(raw.height, math.ceil(cy+side/2)))
                crop = raw.crop(bounds); plain = out/'review'/(ann.stem+'_unmarked.png'); crop.save(plain)
                labelled = crop.copy(); ImageDraw.Draw(labelled).line(
                    [(x-bounds[0], y-bounds[1]) for x, y in poly+[poly[0]]], fill='#00d040', width=2)
                y0 = 36+row*255
                for col, tile in enumerate((crop, labelled)):
                    thumb = ImageOps.contain(tile, (465, 225)); page.paste(thumb, (10+480*col, y0))
                drawing.text((10, y0+227), ann.stem+' crop origin '+str(bounds[:2]), fill='black')
            rec = dict(image=ann.stem, role='TRAIN_CONTRACT_REVIEW_ONLY', source_image=str(image),
                       source_annotation=str(ann), crop_bounds_original_px=bounds,
                       plain_review=str(plain.relative_to(out)), existing_polygon=poly)
            review.append(rec)
            templates.append(dict(image=ann.stem, role=rec['role'], review_image=rec['plain_review'],
                crop_x0=bounds[0], crop_y0=bounds[1], object_definition='',
                visible_four_corners_crop_px='', physical_length_m='', physical_width_m='',
                independent_measurement_source='', visibility_or_occlusion='', decision='PENDING'))
        page.save(out/'review'/(group+'_contract.png'))
    for video in video_info:
        total = video['decoded_frame_count']
        page = Image.new('RGB', (960, 600), 'white'); d = ImageDraw.Draw(page)
        d.text((10, 8), video['name']+' '+video['role']+'; no annotations / predictions', fill='black')
        video['review_frames'] = []
        for i, fraction in enumerate((.05, .35, .65, .9)):
            frame = int((total-1)*fraction); dest = out/'review'/(video['name'][:-4].replace(' ', '_')+f'_f{frame:05d}.jpg')
            command(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', video['source'],
                '-vf', 'select=eq(n\\,'+str(frame)+')', '-vsync', '0', '-frames:v', '1', str(dest)])
            with Image.open(dest) as im:
                if im.size != (video['stream']['width'], video['stream']['height']):
                    raise ValueError('Decoded source size differs')
                page.paste(ImageOps.contain(im.convert('RGB'), (470, 255)), (10+480*(i%2), 32+280*(i//2)))
            d.text((10+480*(i%2), 290+280*(i//2)), 'decoded frame '+str(frame), fill='black')
            video['review_frames'].append(dict(frame_index=frame, file=str(dest.relative_to(out))))
        page.save(out/'review'/('video_'+video['name'][3:-5]+'_overview.jpg'))
    with (out/'independent_annotation_review.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(templates[0])); writer.writeheader(); writer.writerows(templates)
    for name, digest in inputs.items():
        if depth.sha(name) != digest:
            raise ValueError('Input changed during review: '+name)
    report = dict(protocol=VERSION, physical_reference=physical, inputs_sha256=inputs,
        video_candidates=video_info, train_review=review, annotation_contract=json.loads(conversion_path.read_text())['k'],
        scope=dict(model_updates=0, inference=False, depth_fitting=False, labels_changed=False,
                   split_changed=False, detection_val_test_images_access=False),
        status='SOURCE_AUDIT_AND_REVIEW_MATERIAL_COMPLETE_MANUAL_EVIDENCE_PENDING',
        note='Source equality is not recording-event independence. Blank human measurements are not GT. '
             'Video previews cannot establish size error or downstream accuracy.')
    depth.write_new(out/'reference_review.json', report)
    depth.write_new(out/'artifacts.json', {str(f.relative_to(out)): depth.sha(f)
        for f in sorted(out.rglob('*')) if f.is_file()})
    print(report['status'], out)


if __name__ == '__main__':
    main()
