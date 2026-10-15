"""Frozen M -> historical pilot203 images; labels opened only AFTER predictions."""
import argparse,csv,gzip,hashlib,json,os,time,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from crane_project.utils import port_pilot_size_support_v1 as core
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'crane_project/tools/port_pilot_size_support_v1_manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check_sources(m):
    for p,h in m['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Source SHA mismatch '+p)
    for p,h in m['input_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen input SHA mismatch '+p)

def checked_data(data,m):
    if sha(data/'split_manifest.json')!=m['split_sha256']:raise ValueError('Split SHA mismatch')
    split=read(data/'split_manifest.json');stems=core.train_stems(split)
    for name,h in m['sampling_metadata_pins'].items():
        if sha(data/name)!=h:raise ValueError('Sampling provenance mismatch '+name)
    if set(m['data_pins'])!={str(Path(folder)/(i+ext)) for i in stems for folder,ext in [('images','.jpg'),('annfiles','.txt'),('axis_json','.json')]}:raise ValueError('Pins must cover203 only')
    for p,h in m['data_pins'].items():
        if sha(data/p)!=h:raise ValueError('Pilot data differs '+p)
    # Image identity and original-frame metadata only. No GT parsed here.
    result=[]
    for i in stems:
        path=data/'images'/(i+'.jpg')
        with Image.open(path) as im:size=list(im.size)
        if size!=[1920,1080]:raise ValueError('Wrong pilot source/real_seq11 name collision')
        result.append(dict(image=i,sequence='real_seq11',frame_id=int(i.rsplit('_',1)[1]),domain='real',image_size=size,image_sha256=sha(path)))
    return result

def run(a):
    if Path.cwd().resolve()!=ROOT:raise ValueError('Run in project root')
    m=read(MANIFEST);check_sources(m);sources=checked_data(a.data_root,m)
    if a.out.exists():raise ValueError('Refuse overwrite')
    a.out.mkdir(parents=True);write(a.out/'protocol.json',m)
    import torch
    from mmdet.datasets.pipelines import Compose
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as e
    from crane_project.utils.port_midpoint_sigma15_reliability_v1 import Sigma15Reliability
    runtime=e.f.runtime(a.gpu);e.f.seed_all();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    headpath=ROOT/m['head'];payload=torch.load(str(headpath),map_location='cpu')
    if (payload['sigma_cells']!=1.5 or payload['epoch']!=3 or payload['updates']!=2706 or payload['head_digest']!=e.FIXED['head_digest'] or payload['frozen_b']!=e.g.ready.FROZEN_B):raise ValueError('Wrong frozen M payload')
    cfg=e.g.check_cfg();detector=e.f.build_detector(cfg,a.gpu)
    if detector.strict_training_checks or detector.bbox_head.strict_training_checks:raise ValueError('Training-only checks unexpectedly enabled')
    if e.g.state_digest(detector)!=payload['context']['detector_state']:raise ValueError('Frozen detector state mismatch')
    head=e.t.model.SigmaMidpointHead(1.5).cuda(a.gpu);head.load_state_dict(payload['head_state'],strict=True);head.eval().requires_grad_(False)
    if e.g.state_digest(head)!=e.FIXED['head_digest']:raise ValueError('Head digest mismatch')
    before=dict(b=e.g.state_digest(detector),head=e.g.state_digest(head))
    policy=read(ROOT/m['policy']);decision=Sigma15Reliability(policy,policy['front_end'])
    if cfg.data.val.pipeline!=cfg.data.test.pipeline:raise ValueError('Original deterministic pipeline differs')
    if 'LoadAnnotations' in json.dumps(cfg.data.val.pipeline):raise ValueError('No labels allowed in online pipeline')
    pipe=Compose(cfg.data.val.pipeline)
    class ImageOnlyViews:
        def __getitem__(self,index):
            return pipe(dict(img_info=dict(filename=str((a.data_root/'images'/(sources[index]['image']+'.jpg')).resolve())),img_prefix=None))
    dataset=ImageOnlyViews();events=Counter();predictions=[]
    def audit(v):events[v['stage']]+=1
    with (a.out/'predictions_gt_free.jsonl').open('x') as f:
        for index,s in enumerate(sources):
            image,meta=e.old.load_view(dataset,index,s,a.gpu)
            value=e.old.capture(detector,head,image,meta,audit=audit)
            d=decision.decide(value['midpoint'],s['image_size']);sd=decision.decide(value['midpoint'],s['image_size'],'score_only')
            assert d['final_box_original']==value['midpoint']==sd['final_box_original']
            assert d['center_accepted']==(value['midpoint'] is not None)
            row=dict(s,pred=value['midpoint'],b_original=value['b'],decision=d,score_decision=sd,midpoint_checks=value)
            predictions.append(row);f.write(json.dumps(row,allow_nan=False)+'\n')
            if (index+1)%25==0 or index==202:print('Frozen M pilot',index+1,'/203',flush=True)
    # Online prediction stream is closed and hashed BEFORE reading any labels.
    streamsha=sha(a.out/'predictions_gt_free.jsonl')
    after=dict(b=e.g.state_digest(detector),head=e.g.state_digest(head));assert after==before
    assert events['test_feature_extracted']==203 and events['test_native_head_called']==609
    from crane_project.tools.eval_crane_offline import parse_dota_txt
    original=read(a.data_root/'original_59_sampling_manifest.json');origids={Path(r['filename']).stem for r in original['records']}
    rows=[];axis_gt_delta=[]
    for p in predictions:
        ann=a.data_root/'annfiles'/(p['image']+'.txt');axis=read(a.data_root/'axis_json'/(p['image']+'.json'))
        if [axis['imageWidth'],axis['imageHeight']]!=p['image_size']:raise ValueError('Axis image size')
        gt=parse_dota_txt(str(ann))
        if len(gt)!=1:raise ValueError('One GT required')
        gt=np.asarray(gt[0],dtype=np.float32).tolist()
        shapes=[s for s in axis['shapes'] if s.get('label')=='axis' and s.get('shape_type')=='line' and len(s['points'])==2]
        if len(shapes)!=1:raise ValueError('One manual axis required')
        axislength=float(np.linalg.norm(np.asarray(shapes[0]['points'][0])-shapes[0]['points'][1]))
        edges=sorted(gt[2:4],reverse=True);delta=[abs(edges[0]-axislength),abs(edges[1]-axislength/1.9)]
        if max(delta)>.3:raise ValueError('Axis/k1.9 vs existing DOTA convention mismatch '+p['image'])
        axis_gt_delta.append(delta);detail=core.detail(p['pred'],gt)
        state='MISSING' if detail is None else ('FA' if detail['wrong'] else 'CR') if p['decision']['size_accepted'] else ('ED' if detail['wrong'] else 'FR')
        rows.append(dict(p,gt=gt,annotation_sha256=sha(ann),detail=detail,size_state=state,split='extra_source_real_seq11_pilot_k1p9_v2_train',reliability_role='train',sampling_origin='original59' if p['image'] in origids else 'supplement192'))
    assert sha(a.out/'predictions_gt_free.jsonl')==streamsha
    with gzip.open(a.out/'scored_PILOT_TRAIN.jsonl.gz','xt') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    result=core.report(rows)
    check_sources(m);checked_data(a.data_root,m)
    result.update(status='PILOT_TRAIN203_SUPPORT_AUDIT_COMPLETE_NO_TRAINING',scope=dict(detector_updates=0,midpoint_updates=0,reliability_updates=0,aux_val_frames=0,VAL_frames=0,TEST_frames=0,new_policy=False,GT_online=False),
                  proof=dict(state_before=before,state_after=after,events=dict(events),GT_free_stream_sha256=streamsha,all_input_hashes_unchanged=True,axis_to_DOTA_max_abs_edge_delta_px=np.max(axis_gt_delta,axis=0).tolist()),runtime=dict(libraries=runtime,cwd=str(ROOT),environment=os.environ.get('CONDA_DEFAULT_ENV'),gpu=a.gpu,HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()),manifest_sha256=sha(MANIFEST))
    write(a.out/'report.json',result)
    with (a.out/'per_frame.csv').open('w',newline='') as f:
        keys=['image','frame_id','sampling_origin','size_state','signed_long','signed_short','axis','direction','severity','center_error_px','angle_error_deg'];w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:
            d={k:r[k] for k in keys[:4]};d.update({k:v for k,v in (r['detail'] or {}).items() if k in keys});w.writerow(d)
    lines=['# 现成pilot203训练身份：冻结M尺寸错误支持','', '|方法|FA|FR|ED|CR|MISSING|','|---|---:|---:|---:|---:|---:|']
    for method,v in result['methods'].items():lines.append('|%s|%d|%d|%d|%d|%d|'%(method,*(v['states'][k] for k in ('FA','FR','ED','CR','MISSING'))))
    lines+=['',json.dumps(result['error_types'],ensure_ascii=False),'','仅历史203 TRAIN身份；未使用48辅助验证、正式VAL/TEST。未拟合任何判断器。连续段不是独立事件，支持数量不等于训练收益。']
    (a.out/'analysis.md').write_text('\n'.join(lines)+'\n')
    write(a.out/'completion.json',dict(status=result['status'],artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}));print(result['status'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'crane_project/data/crane_grab/extra_source_real_seq11_pilot_k1p9_v2');p.add_argument('--out',required=True,type=Path);p.add_argument('--gpu',type=int,default=3);a=p.parse_args();run(a)
