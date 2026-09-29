"""Recompute sim VAL RIoU from frozen DOTA exports; no inference or reselection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crane_project.tools.eval_crane_offline import parse_dota_txt, compute_riou

EPOCHS = (16, 18, 20, 22, 24)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def text_set_hash(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(b'\0')
        h.update(p.read_bytes())
    return h.hexdigest()


def read_boxes(path, prediction=False):
    lines = [s.split() for s in path.read_text().splitlines() if s.strip()]
    for s in lines:
        try:
            if prediction:
                # ckpt_sweep exports eight coordinates followed by score.
                valid = (len(s) == 9 and np.isfinite(float(s[8]))
                         and 0 <= float(s[8]) <= 1)
            else:
                valid = len(s) in (9, 10) and s[8] == 'grab'
            valid = valid and np.isfinite(np.asarray(s[:8], dtype=float)).all()
        except (ValueError, IndexError):
            valid = False
        if not valid:
            raise ValueError('Invalid {} row: {}'.format(
                'prediction' if prediction else 'GT', path))
    boxes = parse_dota_txt(str(path))
    if len(boxes) > 1 or any(not np.isfinite(b).all() or np.any(b[2:4] <= 0) for b in boxes):
        raise ValueError('Expected valid top-1 box: ' + str(path))
    return boxes


def summarize(ious):
    output = [v for v in ious if v is not None]
    return dict(frames=len(ious), output_frames=len(output),
        coverage_pct=100 * len(output) / len(ious),
        mean_riou_all_frames=sum(output) / len(ious),
        mean_riou_output_only=float(np.mean(output)) if output else None,
        riou_ge_05_frames=sum(v >= .5 for v in output))


def audit(gt_dir, sweep):
    files = sorted(p for p in gt_dir.glob('*.txt') if not p.name.startswith('._'))
    sim = [p for p in files if p.stem.startswith('sim_seq10_')]
    if len(files) != 887 or len(sim) != 512:
        raise ValueError('Expected current VAL: 887 total / 512 sim_seq10')
    gt = {}
    for p in sim:
        boxes = read_boxes(p)
        if len(boxes) != 1:
            raise ValueError('Expected one GT: ' + str(p))
        gt[p.stem] = boxes[0]
    selection_path = sweep / 'sweep_results.json'
    selection = json.loads(selection_path.read_text())
    gt_hash = text_set_hash(files)
    if (selection.get('evidence_role') != 'source_val_checkpoint_selection'
            or selection['source_val_annotations_sha256'] != gt_hash):
        raise ValueError('VAL selection/GT mismatch')
    results = []
    for epoch in EPOCHS:
        key = 'epoch_' + str(epoch)
        directory = sweep / key / 'preds'
        export = directory / 'Task1_grab'
        identity = json.loads((directory / 'Task1_grab.provenance.json').read_text())
        pred_identity = json.loads((directory / 'results.pkl.provenance.json').read_text())
        selected = selection['all_checkpoints'][key]
        ids = identity['frame_ids']
        if len(ids) != len(files) or set(ids) != {p.stem for p in files}:
            raise ValueError('Export frame identity mismatch: ' + key)
        if (pred_identity['split'] != 'source_val'
                or pred_identity['annotations_sha256'] != gt_hash
                or pred_identity['config_sha256'] != selection['config_sha256']
                or pred_identity['checkpoint_sha256'] != selected['checkpoint_sha256']):
            raise ValueError('Prediction identity mismatch: ' + key)
        pkl_hash = sha(directory / 'results.pkl')
        if not (pkl_hash == identity['results_pkl_sha256'] == pred_identity['results_pkl_sha256'] == selected['results_pkl_sha256']):
            raise ValueError('PKL hash mismatch: ' + key)
        if text_set_hash([export / (i + '.txt') for i in ids]) != identity['dota_sha256']:
            raise ValueError('Export hash mismatch: ' + key)
        rows = []
        for p in sim:
            boxes = read_boxes(export / p.name, prediction=True)
            rows.append(dict(frame=p.stem, riou=compute_riou(boxes[0], gt[p.stem]) if boxes else None))
        results.append(dict(epoch=epoch, checkpoint_sha256=selected['checkpoint_sha256'],
            **summarize([r['riou'] for r in rows]), rows=rows))
    return dict(protocol='sim_val_epochs_readonly_v1', gt_dir=str(gt_dir), sweep_dir=str(sweep),
        original_selected_checkpoint=selection['selected_checkpoint'], selection_sha256=sha(selection_path),
        gt_sha256=gt_hash, epochs=results,
        limitations=['Historical paths/config files may have been renamed; original recorded hashes are cross-checked without changing them.',
            'Checkpoint bytes are not loaded or rehashed; identity is bound through the original prediction and selection records.',
            'No TEST data, inference, training or original checkpoint reselection.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--sweep-dir', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    out = Path(args.out_json).resolve()
    if out.exists():
        raise FileExistsError(out)
    result = audit(Path(args.gt_dir).resolve(), Path(args.sweep_dir).resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print('Original VAL selection (unchanged):', result['original_selected_checkpoint'])
    print('epoch  outputs/512  coverage%  RIoU(all)  RIoU(outputs)  IoU>=0.5')
    for r in result['epochs']:
        conditional = 'NA' if r['mean_riou_output_only'] is None else f"{r['mean_riou_output_only']:.6f}"
        print(f"{r['epoch']:5d}  {r['output_frames']:3d}/512  {r['coverage_pct']:9.2f}  {r['mean_riou_all_frames']:.6f}  {conditional}  {r['riou_ge_05_frames']}")
    print('Saved:', out)


if __name__ == '__main__':
    main()
