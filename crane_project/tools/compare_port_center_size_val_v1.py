#!/usr/bin/env python3
"""Reuse B/D VAL-selected PKLs for centre/size geometry; never read TEST."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.audit_port_train_val_geometry_v1 import (
    sha, raw_box, make_row, validate_export, parse_dota_txt, paired_report, write_new)
from crane_project.tools.preflight_port_center_size_v1 import check_configs, CONTROL, EXPERIMENT


def load_val(cfg, path, sweep):
    from mmrotate.datasets import build_dataset
    from crane_project.tools.ckpt_sweep import annotation_set_sha256, check_or_record_prediction
    selection = json.loads((sweep / 'sweep_results.json').read_text())
    selected = selection['selected_checkpoint']
    record = selection['all_checkpoints'][selected]
    checkpoint, pkl = Path(record['checkpoint']), Path(record['results_pkl'])
    ann = ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    actual_ann = (Path(cfg.data.val.get('data_root','')) / cfg.data.val.ann_file).resolve()
    if (actual_ann != ann or selection.get('evidence_role') != 'source_val_checkpoint_selection'
            or selection['config_sha256'] != sha(path)
            or Path(selection['selected_path']).resolve() != checkpoint.resolve()
            or record['checkpoint_sha256'] != sha(checkpoint)
            or record['results_pkl_sha256'] != sha(pkl)
            or selection['source_val_annotations_sha256'] != annotation_set_sha256(str(ann))):
        raise ValueError('VAL-selected artifact identity mismatch: ' + str(path))
    check_or_record_prediction(str(path),str(checkpoint),str(pkl),str(ann),'source_val')
    spec = deepcopy(cfg.data.val)
    spec.test_mode = True
    dataset = build_dataset(spec)
    with pkl.open('rb') as stream:
        predictions = pickle.load(stream)
    if len(dataset) != 887 or len(predictions) != len(dataset):
        raise ValueError('Expected 887 ordered VAL predictions')
    rows = []
    for i,pred in enumerate(predictions):
        row = make_row(dataset,i,raw_box(pred),'val')
        txt = pkl.parent / 'Task1_grab' / (row['image'] + '.txt')
        if not txt.exists():
            raise FileNotFoundError(txt)
        validate_export(pred,parse_dota_txt(str(txt)),1)
        rows.append(row)
    if {r['sequence'] for r in rows} != {'real_seq07','real_seq14','sim_seq10'}:
        raise ValueError('Unexpected VAL sequences')
    return rows, dict(selected_epoch=selected, config_sha256=sha(path),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        pkl_sha256=sha(pkl),selection_sha256=sha(sweep / 'sweep_results.json'))


def name_arms(value):
    """Name the two SymEOOD arms explicitly while reusing the geometry math."""
    if isinstance(value,dict):
        return {k.replace('symeood','compensated_d').replace('eood','control_b'):name_arms(v)
                for k,v in value.items()}
    if isinstance(value,list):
        return [name_arms(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arm,stem in [('b','aug_b'),('d','center_size_d')]:
        ap.add_argument('--'+arm+'-sweep',default=
            'work_dirs/crane_symeood_k1_port_day2night_'+stem+'_v1/val_sweep_port_v1')
    ap.add_argument('--out-json',required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    output = Path(args.out_json).resolve()
    if output.exists():
        raise FileExistsError(output)
    b,d = check_configs()
    rows_b,id_b = load_val(b,CONTROL,Path(args.b_sweep).resolve())
    rows_d,id_d = load_val(d,EXPERIMENT,Path(args.d_sweep).resolve())
    if id_b['selected_epoch'] != 'epoch_24':
        raise ValueError('Control must be the already frozen B epoch_24')
    report = dict(protocol='port_center_size_compensation_val_compare_v1',
        evidence_role='source_val_only_optimization_verification',
        identities=dict(control_b=id_b,compensated_d=id_d),
        geometry=name_arms(paired_report(rows_b,rows_d,True)))
    churn = []
    for a,c in zip(rows_b,rows_d):
        ma,mc = a['metrics'],c['metrics']
        if (ma['output'] != mc['output'] or ma['center_hit'] != mc['center_hit']
                or (ma['riou']>=.5) != (mc['riou']>=.5)):
            churn.append(dict(image=a['image'],control_b=ma,compensated_d=mc))
    report['changed_correctness_rows'] = churn
    report['control_wrong_location_followup'] = [dict(image=a['image'],
        control_b=a['metrics'],compensated_d=c['metrics']) for a,c in zip(rows_b,rows_d)
        if a['metrics']['output'] and a['metrics']['riou'] < .5]
    report['limitations'] = [
        'A complete B/D treatment comparison does not isolate classification ranking from box regression.',
        'Shared FPN updates may change classification and angles despite no direct extra angle/classification gradient.',
        'No TEST loading, threshold changes or checkpoint reselection. Near-square and counterfactual caveats apply.']
    write_new(output,report)
    for arm in ['control_b','compensated_d']:
        brief = {}
        for domain in ['real','sim']:
            summary = report['geometry'][arm+'_strata']['domain:'+domain]
            brief[domain] = dict(frames=summary['frames'],outputs=summary['output_frames'],
                output_center_hit_pct=summary['output_center_hit_pct'],
                all_frame_center_hit_pct=summary['all_frame_center_hit_pct'],
                mean_riou=summary['all_frame_mean_riou'],
                mean_center_px=summary['center_error_px']['mean'],
                mean_long_edge_relative_error=summary['long_edge_relative_error']['mean'],
                mean_short_edge_relative_error=summary['short_edge_relative_error']['mean'],
                pure_angle_rmse=summary['angle_error_deg']['rmse'])
        print(arm, json.dumps(brief,ensure_ascii=False,indent=2))
    print('Detailed VAL comparison:',output)


if __name__ == '__main__':
    main()
