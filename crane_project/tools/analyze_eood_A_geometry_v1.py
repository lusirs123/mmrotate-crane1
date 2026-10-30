"""CPU Top1 readback of all five fixed E+A VAL epochs; no TEST or selection change."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import ap11, validate_saved_gt
    from crane_project.tools.analyze_symeood_ablation_first4_v1 import txtbox
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical, row_metrics
    from crane_project.utils.eood_A_geometry_protocol_v1 import assess_detection
    base = args.base.resolve()
    c = json.loads((base/'launch_protocol_gpus23.json').read_text())
    sweep = json.loads((base/'train/ckpt_sweep/sweep_results.json').read_text())
    assert sweep['selection_config'] == c['checkpoint_selection']['selection_config']
    assert sweep['candidate_epochs'] == [16,18,20,22,24]
    selected = sweep['selected_checkpoint']
    reference = ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr/fixed_b/epoch_01_val.rows.jsonl.gz'
    assert sha(reference) == '06f47896881a1555f00d1fde777c28c91f2dd3e0800d3b667e6e5c20eae7fc7b'
    with gzip.open(reference,'rt') as f:
        refs = [json.loads(line) for line in f]
    assert len(refs)==887 and [r['image'] for r in refs]==sorted(r['image'] for r in refs)
    for r in refs:
        path = ROOT/'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'/(r['image']+'.txt')
        poly = np.asarray([float(v) for v in path.read_text().split()[:8]]).reshape(4,2)
        validate_saved_gt(r['gt'],poly,txtbox(path))
    out = base/'VAL_analysis'
    out.mkdir(exist_ok=False)
    frames, metrics, sources = [], [], {}
    for epoch in (16,18,20,22,24):
        name = 'epoch_%d'%epoch
        cp = sweep['all_checkpoints'][name]
        prediction = Path(cp['results_pkl'])
        assert sha(prediction)==cp['results_pkl_sha256']
        with prediction.open('rb') as f:
            raw = pickle.load(f)
        assert len(raw)==887
        sources[str(prediction)]=sha(prediction)
        rows=[]
        for ref, result in zip(refs,raw):
            assert len(result)==1
            boxes=np.asarray(result[0])
            assert boxes.ndim==2 and boxes.shape[1]==6 and np.isfinite(boxes).all()
            pred=boxes[boxes[:,5].argmax()].tolist() if len(boxes) else None
            riou=compute_riou(np.asarray(canonical(pred)),np.asarray(canonical(ref['gt']))) if pred else 0.
            row={k:ref[k] for k in ('image','sequence','domain','frame_id','gt')}
            row.update(epoch=name,selected=name==selected,prediction=pred,score=pred[5] if pred else 0.)
            row.update(row_metrics(ref['gt'],pred,riou))
            rows.append(row)
        frames.extend(rows)
        for domain in ('all','real','sim'):
            group=[r for r in rows if domain=='all' or r['domain']==domain]
            valid=[r for r in group if r['output']]
            assert valid
            value=dict(side='E_plus_A',epoch=name,selected=name==selected,domain=domain,
                frames=len(group),outputs=len(valid),TP50=sum(r['riou']>=.5 for r in group),
                TP75=sum(r['riou']>=.75 for r in group),
                Recall_pct=100*sum(r['riou']>=.5 for r in group)/len(group),
                center_MAE_px=sum(r['center_error_px'] for r in valid)/len(valid),
                mean_RIoU=sum(r['riou'] for r in group)/len(group),
                Sim_angle_RMSE_deg=math.sqrt(sum(r['angle_error_deg']**2 for r in valid)/len(valid)) if domain=='sim' else None)
            order=np.argsort(-np.asarray([r['score'] for r in valid],dtype=np.float32))
            for threshold in (.5,.75):
                tp=np.asarray([r['riou']>=threshold for r in valid],dtype=np.float32)[order].cumsum()
                fp=np.asarray([r['riou']<threshold for r in valid],dtype=np.float32)[order].cumsum()
                precision=tp/np.maximum(tp+fp,np.finfo(np.float32).eps)
                value['AP'+str(round(threshold*100))]=100*ap11(tp.astype(np.float64)/len(group),precision)
            metrics.append(value)
    chosen=next(r for r in metrics if r['selected'] and r['domain']=='all')
    assessment=assess_detection(chosen,c['baseline']['baseline_VAL'])
    with (out/'metrics.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with (out/'per_frame.jsonl').open('x') as f:
        for r in frames:f.write(json.dumps(r,allow_nan=False)+'\n')
    result=dict(passed=True,model_forwards=0,optimizer_updates=0,TEST=False,selection_changed=False,
        selected_checkpoint=selected,baseline=c['baseline']['baseline_VAL'],selected=chosen,
        detection_gate=assessment,metrics=metrics,sources=sources,
        strict_initialization_pair=False,automatic_adoption=False,
        GT_reference_sha256=sha(reference),source_sha256=sha(__file__))
    with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('E_A_FIXED_VAL_READBACK_COMPLETE',selected,flush=True)


if __name__=='__main__':
    main()
