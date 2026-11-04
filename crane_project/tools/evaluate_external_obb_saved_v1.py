"""Reuse existing FAA/common metrics on identity-bound external saved results.

Runs CPU metrics only. No detector inference, training or checkpoint selection.
"""
import argparse
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import DATA, read, sha, dataset_manifest, check_sources


def validate_export(directory):
    from crane_project.tools.export_external_obb_v1 import config_identity
    directory = Path(directory)
    task, receipt = read(directory / 'task.json'), read(directory / 'export_receipt.json')
    if not receipt['passed'] or receipt['training_updates'] != 0:
        raise ValueError('Incomplete external export')
    if receipt['model'] != task['model'] or receipt['split'] != task['split']:
        raise ValueError('Task/export identity differs')
    if receipt['configuration'] != config_identity(task['model']):
        raise ValueError('Model/export configuration changed')
    if receipt['sources'] != check_sources():
        raise ValueError('Official sources changed')
    rows = [r for r in dataset_manifest() if r['split'] == task['split']]
    for row in rows:
        if sha(DATA / row['annfiles']['path']) != row['annfiles']['sha256']:
            raise ValueError('Evaluation annotation bytes changed')
    ids = sorted(r['id'] for r in rows)
    if task['frame_ids'] != ids or receipt['frame_ids'] != ids or receipt['frames'] != len(ids):
        raise ValueError('Frame identities differ from sealed split')
    if Path(task['annotations']).resolve() != DATA / task['split'] / 'annfiles':
        raise ValueError('Ground-truth directory differs')
    if sha(directory / 'preds/results.pkl') != receipt['results_sha256']:
        raise ValueError('Saved predictions changed')
    if sha(task['checkpoint']) != receipt['checkpoint_sha256'] or task['checkpoint_sha256'] != receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint identity differs')
    return task, receipt


def evaluate(directory):
    from crane_project.tools.eval_faa_val_test_v1 import paper_metrics
    from crane_project.tools.ckpt_sweep import pkl_to_dota
    directory = Path(directory)
    task, receipt = validate_export(directory)
    with (directory / 'preds/results.pkl').open('rb') as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(task['frame_ids']):
        raise ValueError('Prediction count differs')
    pkl_to_dota(str(directory / 'preds/results.pkl'), task['frame_ids'], str(directory / 'preds'))
    paper_metrics(directory, task['split'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    evaluate(args.directory.resolve())


if __name__ == '__main__':
    main()
