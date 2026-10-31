"""CPU verification of completed VAL: native AP, saved bags, paired changes.

No model construction/forward, optimizer, training, TEST or checkpoint selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    import torch
    from mmrotate.core.evaluation.eval_map import eval_rbbox_map
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import ap11
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical
    torch.set_num_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    base = args.base.resolve()
    summary = read(base/'VAL_analysis/summary.json')
    sweep = read(base/'train/ckpt_sweep/sweep_results.json')
    assert summary['selected_checkpoint'] == sweep['selected_checkpoint'] == 'epoch_22'
    oldbase = ROOT/'work_dirs/crane_eood_k1_port_day2night_aug_b_v1'
    oldsweep = read(oldbase/'val_sweep_port_v1/sweep_results.json')
    assert oldsweep['selected_checkpoint'] == 'epoch_24'
    oldrows = [json.loads(line) for line in (
        ROOT/'work_dirs/symeood_b2_20261030_v1/source_audit_v2/per_frame.jsonl').read_text().splitlines()]
    oldrows = [r for r in oldrows if r['side']=='E' and r['split']=='val']
    newrows = [json.loads(line) for line in (base/'VAL_analysis/per_frame.jsonl').read_text().splitlines()]
    assert len(oldrows)==887 and len(newrows)==4435
    byname={r['image']:r for r in oldrows}
    identities=[r['image'] for r in oldrows]
    assert identities==sorted(identities)
    checks, decompositions, bag_results, comparisons = [], [], [], []

    def curve(rows):
        valid=[r for r in rows if r['output']]
        order=np.argsort(-np.asarray([r['score'] for r in valid],dtype=np.float32))
        tp=np.asarray([r['riou']>=.75 for r in valid],dtype=np.float32)[order].cumsum()
        fp=np.asarray([r['riou']<.75 for r in valid],dtype=np.float32)[order].cumsum()
        recall=tp.astype(np.float64)/len(rows)
        precision=tp/np.maximum(tp+fp,np.finfo(np.float32).eps)
        values=[]
        for t in np.arange(0,1.001,.1):
            v=precision[recall>=t]
            values.append(float(v.max()) if v.size else 0.)
        return dict(AP75=100*ap11(recall,precision),recall_points=list(np.arange(0,1.001,.1)),
            interpolated_precision=values,AP_contributions_pp=[v*100/11 for v in values],
            achieved_recall75=float(recall[-1]),
            top100_low75=sum(valid[int(i)]['riou']<.75 for i in order[:100]))

    def selected_rows(epoch):
        rows=[r for r in newrows if r['epoch']==epoch]
        assert len(rows)==887 and [r['image'] for r in rows]==identities
        assert all(r['gt']==byname[r['image']]['gt'] for r in rows)
        return rows

    for arm,rows in [('E',oldrows)]+[(e,selected_rows(e)) for e in ['epoch_16','epoch_18','epoch_20','epoch_22','epoch_24']]:
        for domain in ('all','real','sim'):
            group=[r for r in rows if domain=='all' or r['domain']==domain]
            data=curve(group)
            decompositions.append(dict(arm=arm,domain=domain,**data))
            if arm!='E':
                metric=next(r for r in summary['metrics'] if r['epoch']==arm and r['domain']==domain)
                assert abs(data['AP75']-metric['AP75'])<1e-5
            common=[r for r in group if r['output'] and byname[r['image']]['output']]
            recover=sum(r['riou']>=.75 and byname[r['image']]['riou']<.75 for r in group)
            lose=sum(r['riou']<.75 and byname[r['image']]['riou']>=.75 for r in group)
            comparisons.append(dict(arm=arm,domain=domain,TP75_recovered=recover,TP75_lost=lose,
                common_outputs=len(common),old_common_center_MAE=sum(byname[r['image']]['center_error_px'] for r in common)/len(common),
                new_common_center_MAE=sum(r['center_error_px'] for r in common)/len(common)))

    for arm,cp,rows in [('E',oldsweep['all_checkpoints']['epoch_24'],oldrows)]+[
        (e,sweep['all_checkpoints'][e],selected_rows(e)) for e in ['epoch_20','epoch_22','epoch_24']]:
        path=Path(cp['results_pkl'])
        assert sha(path)==cp['results_pkl_sha256']
        with path.open('rb') as f:raw=pickle.load(f)
        assert len(raw)==887
        saved_bag=[];top1=[]
        for result,row in zip(raw,rows):
            boxes=np.asarray(result[0]);assert np.isfinite(boxes).all()
            best=boxes[boxes[:,5].argmax()] if len(boxes) else None
            assert (best is not None)==row['output']
            if best is not None:
                assert np.allclose(best,row['prediction'],atol=1e-3,rtol=1e-6)
            ious=[float(compute_riou(np.asarray(canonical(v)),np.asarray(canonical(row['gt'])))) for v in boxes]
            saved_bag.append(dict(image=row['image'],domain=row['domain'],count=len(boxes),
                has_good=any(v>=.75 for v in ious),native_good=row['riou']>=.75))
            top1.append([best[None,:].astype(np.float32) if best is not None else np.empty((0,6),np.float32)])
        for domain in ('all','real','sim'):
            group=[r for r in saved_bag if domain=='all' or r['domain']==domain]
            bag_results.append(dict(arm=arm,domain=domain,frames=len(group),candidate_count=sum(r['count'] for r in group),
                native_TP75=sum(r['native_good'] for r in group),
                saved_bag_oracle_TP75=sum(r['has_good'] for r in group),
                recoverable_by_saved_bag_ranking=sum(r['has_good'] and not r['native_good'] for r in group),
                no_good_in_saved_bag=sum(not r['has_good'] for r in group)))
        if arm in ('E','epoch_22'):
            anns=[dict(bboxes=np.asarray([r['gt']],np.float32),labels=np.zeros(1,np.int64),
                       bboxes_ignore=np.empty((0,5),np.float32),labels_ignore=np.empty(0,np.int64)) for r in rows]
            for threshold in (.5,.75):
                ap,_=eval_rbbox_map(top1,anns,iou_thr=threshold,use_07_metric=True,dataset=('grab',),logger='silent',nproc=1)
                reference=(summary['baseline'] if arm=='E' else summary['selected'])['AP'+str(round(100*threshold))]
                delta=abs(ap*100-reference)
                assert delta<1e-4,(arm,threshold,ap*100,reference)
                checks.append(dict(arm=arm,threshold=threshold,native_AP=ap*100,stored_AP=reference,delta=delta))
    assert not torch.cuda.is_initialized()
    args.out.mkdir(parents=True,exist_ok=False)
    output=dict(passed=True,native_AP_checks=checks,AP75_decomposition=decompositions,
        saved_post_NMS_bag=bag_results,paired_frame_changes=comparisons,
        model_forwards=0,optimizer_updates=0,TEST=False,selection_changed=False,
        limitations=['Saved bags are after score filtering/top-k/NMS; not all dense candidates.',
                    'No causal root of TRAIN assignment is established by these VAL readbacks.',
                    'Eleven-point contributions explain AP arithmetic; they do not remove a real high-IoU loss.'],
        source_sha256=sha(__file__))
    (args.out/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,native_AP_checks=checks,
        selected_bag=[r for r in bag_results if r['arm'] in ('E','epoch_22')],
        selected_pairing=[r for r in comparisons if r['arm']=='epoch_22']),indent=2),flush=True)


if __name__=='__main__':
    main()
