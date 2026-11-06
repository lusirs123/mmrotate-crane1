"""Select a fixed external checkpoint set using the existing source-VAL rule.

Consumes completed VAL exports. No training, inference or automatic TEST.
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import DATA, MANIFEST_SHA, YOLO_CONFIG, read, sha, write_new


def select(model, val_dir, out):
    from crane_project.tools import ckpt_sweep as sweep
    from crane_project.tools.evaluate_external_obb_saved_v1 import validate_export
    from crane_project.tools.export_external_obb_v1 import config_identity
    cfg = read(ROOT / (YOLO_CONFIG if model == 'yolo26' else 'configs/gaucho/experiment.json'))
    candidates = cfg['selection']['candidate_completed_epochs']
    native_names = cfg['selection']['native_checkpoint_names']
    records = OrderedDict()
    configuration = config_identity(model)
    for epoch, native_name in zip(candidates, native_names):
        directory = val_dir / ('epoch_%d' % epoch)
        task, receipt = validate_export(directory)
        if task['model'] != model or task['split'] != 'val' or receipt['configuration'] != configuration:
            raise ValueError('Only the declared model/source-VAL exports may select weights')
        if Path(task['checkpoint']).name != native_name:
            raise ValueError('Checkpoint epoch/name differs from preparation contract')
        task_dir = sweep.pkl_to_dota(str(directory / 'preds/results.pkl'), task['frame_ids'], str(directory / 'preds'))
        metrics = sweep.run_offline_eval(task_dir, str(DATA / 'val/annfiles'), center_thresh=15.)
        records['epoch_%d' % epoch] = dict(checkpoint=task['checkpoint'],
            checkpoint_sha256=receipt['checkpoint_sha256'], metrics=metrics,
            export_receipt=str(directory / 'export_receipt.json'),
            export_receipt_sha256=sha(directory / 'export_receipt.json'))
    selected, data, info = sweep.select_best_checkpoint(records, sweep.SELECTION_CONFIG)
    if selected is None:
        raise ValueError('No source-VAL checkpoint could be selected')
    out.mkdir(parents=True, exist_ok=False)
    result_path = out / 'sweep_results.json'
    write_new(result_path, dict(model=model, evidence_role='source_val_checkpoint_selection',
        selection_config=sweep.SELECTION_CONFIG, selection_rule_source_sha256=sha(Path(sweep.__file__)),
        candidate_completed_epochs=candidates, selection_info=info,
        center_thresh_px=15.0,
        selected_checkpoint=selected, selected_path=data['checkpoint'], all_checkpoints=records,
        configuration=configuration, manifest_sha256=MANIFEST_SHA, TEST_used_for_selection=False))
    write_new(out / 'selected_identity.json', dict(model=model,
        evidence_role='source_val_checkpoint_selection', checkpoint=data['checkpoint'],
        checkpoint_sha256=data['checkpoint_sha256'], selected_completed_epoch=int(selected.split('_')[-1]),
        configuration=configuration, manifest_sha256=MANIFEST_SHA,
        selection_results=str(result_path.resolve()), selection_results_sha256=sha(result_path),
        locked_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        TEST_used_for_selection=False, auto_TEST=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('gaucho', 'yolo26'), required=True)
    parser.add_argument('--val-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    select(args.model, args.val_dir.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
