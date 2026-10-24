"""Reaggregate frozen C observation files on CPU, no model/torch imports."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


GROUPS=('100','101','110','111')
CATEGORIES=('no_qualified','score_filter_lost','ranking_lost','success')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def stats(v):
    return dict(mean=statistics.mean(v),median=statistics.median(v),minimum=min(v),maximum=max(v),count=len(v)) if v else None


def csv_write(path,rows):
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def main(source,out):
    import numpy as np
    import cv2
    complete=json.loads((source/'complete.json').read_text());assert complete['passed'] and complete['updates']==0
    draft=json.loads((source/'draft.json').read_text())
    root=source.parents[2]
    for name,digest in draft['sources'].items():assert sha(root/name)==digest
    for rec in draft['groups'].values():
        for field,h in [('weight','weight_sha256'),('config','config_sha256'),('val_pkl','val_pkl_sha256')]:
            assert sha(Path(rec[field]))==rec[h]
    state=read_jsonl(source/'state_checks.jsonl')
    state.extend(json.loads(f.read_text()) for f in source.glob('reader_*.json'))
    state_map={(r['group'],r['phase']):r for r in state}
    assert len(state_map)==12
    for g in GROUPS:
        hashes=[]
        for phase in ('val','train','gradient'):
            r=state_map[g,phase];assert r['passed'] and r['before']==r['after'];hashes.append(r['before'])
        assert len(set(hashes))==1
    for phase,n in [('val',887),('train',2558)]:
        receipts=[json.loads((source/g/('candidates_'+phase+'_receipt.json')).read_text()) for g in GROUPS]
        assert all(r['passed'] and r['frames']==n for r in receipts)
        assert len({r['input_tensor_sequence_sha256'] for r in receipts})==1
    # Identity recheck includes the image/GT bytes and the prepared batch data.
    for item in draft['data_inventory']:
        assert sha(Path(item['image']))==item['image_sha256']
        assert sha(Path(item['annotation']))==item['annotation_sha256']
    for view in draft['training_views']:assert sha(Path(view['path']))==view['sha256']
    out.mkdir(exist_ok=False,parents=True)
    candidate_rows=[];mu_rows=[];gradient_rows=[];label_rows=[];mu_counter=collections.defaultdict(lambda:collections.defaultdict(float))
    verification=collections.Counter()
    inventory={r['frame']:r for r in draft['data_inventory']}
    sequence_rows=[]
    for g in GROUPS:
        for split in ('val','train'):
            rs=read_jsonl(source/g/('candidates_'+split+'_summary.jsonl'))
            assert len(rs)==(887 if split=='val' else 2558)
            for r in rs:
                f=source/g/('candidates_'+split)/(r['frame']+'.npz')
                assert sha(f)==r['candidate_file_sha256']
                with np.load(f) as z:
                    score=z['score'];iou=z['iou'];passed=score>.05;qualified=iou>=.5
                    for field in ('score','iou','center','angle','box'):assert np.isfinite(z[field]).all(),(g,r['frame'],field)
                    assert (iou>=0).all() and (iou<=1.0001).all()
                    selected=int(z['selected'])
                    assert np.array_equal(passed,z['passed'])
                    assert r['candidates']==len(iou) and r['qualified']==int(qualified.sum())
                    assert r['qualified_passed']==int((qualified&passed).sum())
                    cat='no_qualified' if not qualified.any() else 'score_filter_lost' if not (qualified&passed).any() else 'success' if selected>=0 and qualified[selected] else 'ranking_lost'
                    assert cat==r['category']
                    assert selected<0 or score[selected]==score[passed].max()
                r['group']=g;r['split']=split;r['domain']=r['frame'].split('_')[0]
                r['seq']='_'.join(r['frame'].split('_')[:-1])
                poly=np.array([float(x) for x in Path(inventory[r['frame']]['annotation']).read_text().split()[:8]],np.float32).reshape(4,2)
                _,(w,h),_=cv2.minAreaRect(poly)
                r['angle_eligible']=max(w,h)/min(w,h)>=1.2
                verification['candidate_files']+=1
            for domain in ('real','sim','all'):
                subset=[r for r in rs if domain=='all' or r['domain']==domain]
                c=collections.Counter(r['category'] for r in subset)
                assert sum(c.values())==len(subset)
                outputs=[r for r in subset if r['selected']>=0]
                candidate_rows.append(dict(group=g,split=split,domain=domain,frames=len(subset),outputs=len(outputs),
                    **{k:c[k] for k in CATEGORIES},qualified_available=sum(r['qualified']>0 for r in subset),
                    center_correct=sum(r['selected_center']<15 for r in outputs),
                    angle_eligible=sum(r['angle_eligible'] for r in outputs),
                    angle_correct=sum(r['angle_eligible'] and r['selected_angle']<=3 for r in outputs)))
            for seq in sorted({r['seq'] for r in rs}):
                subset=[r for r in rs if r['seq']==seq];c=collections.Counter(r['category'] for r in subset)
                sequence_rows.append(dict(group=g,split=split,seq=seq,frames=len(subset),**{k:c[k] for k in CATEGORIES}))
        grad=read_jsonl(source/g/'gradient_summary.jsonl');assert len(grad)==128
        for rec,view in zip(grad,draft['training_views']):
            assert rec['batch']==view['batch']
            assert rec['frames']==[r['frame'] for r in view['samples']]
            if rec['strength_degenerate']:
                verification['degenerate_batches']+=1;continue
            assert math.isclose(rec['S'],rec['G'],rel_tol=1e-6,abs_tol=1e-8)
            verification['matched_loss_batches']+=1
            per_losses_U=per_losses_G=0.
            raw_squared={k:[0.,0.,0.] for k in ('U','G','S')}
            raw_GS_dot=0.
            for level,summary in enumerate(rec['levels']):
                file=source/g/'loss_samples'/('batch_%03d_level%d.npz'%(rec['batch'],level))
                with np.load(file) as z:
                    label=z['labels'];weight=z['label_weight'];top=z['topk'];mu=z['mu'];u=z['loss_U'].astype(np.float64);gg=z['loss_G'].astype(np.float64)
                    own=z['own_reference_mu_same_topk']
                    assert len(mu)==summary['candidates'] and (mu>=1).all() and (mu<=2).all()
                    assert np.all(own<=mu+1e-6)
                    pos=(label==0)&(weight>0);neg=(label!=0)&(weight>0);ignore=weight<=0
                    assert int(top.sum())==summary['positive_topk']+summary['negative_topk']+summary['ignored_topk']
                    per_losses_U+=float(u.sum());per_losses_G+=float(gg.sum())
                    owner=z['image_ids'];ref=z['reference_gt_index'];gt_images=z['reference_gt_image']
                    actual_external=np.zeros(len(label),dtype=bool)
                    actual_external[top]=gt_images[ref[top]]!=owner[top]
                    for domain in ('real','sim','all'):
                        dm=np.array([view['samples'][int(i)]['frame'].startswith(domain+'_') for i in owner]) if domain!='all' else np.ones(len(label),bool)
                        for name,mask in [('positive',pos),('negative',neg),('ignored',ignore)]:
                            for lev in (level,'all'):
                                target=mu_counter[g,domain,lev,name]
                                m=dm&mask;t=m&top;count=int(t.sum())
                                target['candidates']+=int(m.sum());target['topk']+=count
                                target['loss_U']+=float(u[m].sum());target['loss_G']+=float(gg[m].sum())
                                target['loss_increment']+=float((gg[m]-u[m]).sum())
                                target['same_topk_own_reference_loss_G']+=float((u[m]*own[m]).sum())
                                target['mu_changed_by_own_reference']+=int(((mu-own>1e-6)&t).sum())
                                target['mu_difference_sum']+=float((mu[t]-own[t]).astype(np.float64).sum())
                                target['actual_external_argmin']+=int((actual_external&t).sum())
                                for kind in ('kld','distance'):
                                    for c in range(5):
                                        reference_mask=(z['cross_'+kind]==c)&t
                                        target[kind+'_reference_'+str(c)]+=int(reference_mask.sum())
                                        target[kind+'_reference_'+str(c)+'_loss_increment']+=float((gg[reference_mask]-u[reference_mask]).sum())
                                target['mu_near1']+=int(((mu<1.1)&t).sum());target['mu_near2']+=int(((mu>=1.9)&t).sum())
                                target['mu_middle']+=int(((mu>=1.1)&(mu<1.9)&t).sum())
                    for name,m in [('positive',pos),('negative',neg),('ignored',ignore)]:
                        assert math.isclose(float(u[m].sum()),summary[name+'_loss_U'],rel_tol=1e-5,abs_tol=1e-7)
                        assert math.isclose(float(gg[m].sum()),summary[name+'_loss_G'],rel_tol=1e-5,abs_tol=1e-7)
                    masks=[pos,neg,ignore]
                with np.load(source/g/'loss_samples'/('batch_%03d_level%d_logits_grad.npz'%(rec['batch'],level))) as z:
                    for key in ('U','G','S'):
                        flat=z[key].transpose(0,2,3,1).reshape(-1)
                        assert np.isfinite(flat).all()
                        assert (flat[masks[2]]==0).all()
                        for kind,mask in enumerate(masks):raw_squared[key][kind]+=float(np.square(flat[mask].astype(np.float64)).sum())
                    gu=z['U'].astype(np.float64);gs=z['S'].astype(np.float64)
                    assert np.allclose(gs,gu*rec['alpha'],rtol=1e-4,atol=1e-8)
                    raw_GS_dot+=float((z['G'].astype(np.float64)*gs).sum())
                verification['mu_and_logit_files']+=1
            assert math.isclose(per_losses_U,rec['U'],rel_tol=1e-5,abs_tol=1e-7)
            assert math.isclose(per_losses_G,rec['G'],rel_tol=1e-5,abs_tol=1e-7)
            for index,kind in enumerate(('positive','negative','ignored')):
                for key in ('U','G','S'):
                    assert math.isclose(math.sqrt(raw_squared[key][index]),rec['label_partition'][kind][key],rel_tol=1e-9,abs_tol=1e-12)
            ng=sum(raw_squared['G']);ns=sum(raw_squared['S'])
            raw_cos=raw_GS_dot/math.sqrt(ng*ns)
            assert math.isclose(raw_cos,rec['gradient']['logits']['G_vs_S']['cosine'],rel_tol=1e-9,abs_tol=1e-12)
        valid=[r for r in grad if not r['strength_degenerate']]
        for kind in ('positive','negative','ignored'):
            r=dict(group=g,kind=kind,batches=len(valid))
            for key in ('U','G','S'):
                r[key+'_L2_mean']=statistics.mean(x['label_partition'][kind][key] for x in valid)
                energy=sum(x['label_partition'][kind][key]**2 for x in valid)
                total=sum(x['label_partition'][k][key]**2 for x in valid for k in ('positive','negative','ignored'))
                r[key+'_energy_share_pct']=100*energy/total if total else None
            label_rows.append(r)
        for space in ('logits','cls_head','FPN_features','FPN_parameters','all_parameters'):
            cos=[r['gradient'][space]['G_vs_S']['cosine'] for r in valid if r['gradient'][space]['G_vs_S']['cosine'] is not None]
            ratio=[r['gradient'][space]['G_vs_S']['ratio'] for r in valid if r['gradient'][space]['G_vs_S']['ratio'] is not None]
            gradient_rows.append(dict(group=g,space=space,batches=len(valid),alpha_mean=statistics.mean(r['alpha'] for r in valid),
                alpha_min=min(r['alpha'] for r in valid),alpha_max=max(r['alpha'] for r in valid),
                cosine_mean=statistics.mean(cos),cosine_min=min(cos),ratio_mean=statistics.mean(ratio),ratio_min=min(ratio),ratio_max=max(ratio),
                U_preclip_mean=statistics.mean(r['total']['U']['preclip_norm'] for r in valid),
                G_preclip_mean=statistics.mean(r['total']['G']['preclip_norm'] for r in valid),
                S_preclip_mean=statistics.mean(r['total']['S']['preclip_norm'] for r in valid),
                U_clipped=sum(r['total']['U']['clip_coefficient']<1 for r in valid),
                G_clipped=sum(r['total']['G']['clip_coefficient']<1 for r in valid),
                S_clipped=sum(r['total']['S']['clip_coefficient']<1 for r in valid)))
    for (g,domain,level,kind),counter in mu_counter.items():
        row=dict(group=g,domain=domain,level=level,kind=kind,**dict(counter))
        row['top_mu_reference_delta_mean']=row['mu_difference_sum']/row['topk'] if row['topk'] else None
        assert sum(row['kld_reference_'+str(c)] for c in range(5))==row['topk']
        mu_rows.append(row)
    csv_write(out/'candidate_failures.csv',candidate_rows)
    csv_write(out/'candidate_by_video.csv',sequence_rows)
    csv_write(out/'mu_objects.csv',mu_rows)
    csv_write(out/'strength_matched_gradients.csv',gradient_rows)
    csv_write(out/'label_gradients.csv',label_rows)
    summary=dict(verification=dict(verification),candidates=candidate_rows,mu=[r for r in mu_rows if r['level']=='all'],
                 gradients=gradient_rows,label_gradients=label_rows,empty_gt_calls={g:sum(level['empty_gt'] for r in read_jsonl(source/g/'gradient_summary.jsonl') for level in r['levels']) for g in GROUPS},
                 scope='mature fixed checkpoints and diagnostic batches, no training efficacy claim')
    (out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/'provenance.json').write_text(json.dumps(dict(source=str(source),protocol_sha256=sha(source/'protocol.json'),
         draft_sha256=sha(source/'draft.json'),script_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(summary['verification'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.source.resolve(),a.out.resolve())
