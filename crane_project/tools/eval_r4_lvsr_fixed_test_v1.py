"""One authorized TEST of frozen B-conditioned LVSR; no midpoint execution."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import gzip
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import train_r4_lvsr_without_midpoint_v1 as training
from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
read,save=training.read,training.save
CONFIG=ROOT/'crane_project/configs/crane_r4_lvsr_fixed_test_v1.json'
RUN=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/r4_lvsr_001'
EXTRA=['crane_project/configs/crane_r4_lvsr_fixed_test_v1.json',
 'crane_project/tools/eval_r4_lvsr_fixed_test_v1.py','tests/test_r4_lvsr_fixed_test_v1.py',
 'crane_project/tools/eval_lvsr_partial_visual_norm_fixed_test_v1.py',
 'crane_project/tools/eval_symeood_r4_lvsr_ep5_fixed_test_v1.py',
 'crane_project/tools/eval_symeood_r4_midpoint_fixed_test_v1.py',
 'crane_project/tools/train_symeood_r4_midpoint_v1.py',
 'crane_project/utils/port_geometry_refine_g_v1.py']
L5_ROWS_SHA='9a11d0b625f0a05e45d7f7bacd7e41ff60def9bcd1f31561d30bd412b64ad240'
SPECS={'R4':'b','R4_LB':'edge_residual','R4_M20':'M20','R4_M20_LM':'LM'}


def rows(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]


def status(stage,**details):
    value=dict(stage=stage,pid=os.getpid(),time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        TEST=True,training_updates=0,midpoint_forwards=0,**details)
    tmp=RUN/'test_LB_status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(RUN/'test_LB_status.json')


def checked_config():
    c=read(CONFIG)
    if (c['physical_gpu']!=3 or c['epoch']!=3 or c['TEST_frames']!=1440 or c['batch_size']!=1
        or c['precision']!='FP32' or c['efficiency'] or c['selection_on_test'] or c['automatic_adoption']
        or c['checkpoint_sha256']!='df30cf4d12db56e1c3af0643c25dfd925c733dcd106dcdfc3fc0eec2fc94e4a4'):
        raise ValueError('Frozen LB TEST scope differs')
    return c


def validate_training(s,c,selection):
    cfg=checked_config();cp=dict(path='head_epoch_03.pth',epoch=3,sha256=cfg['checkpoint_sha256'])
    if (c['status']!='R4_LB_TRAIN_VAL_COMPLETE' or c['updates']!=15336 or c['epochs']!=24
        or c['checkpoint']!=cp or c['selected_epoch']!=3 or c['condition']!='B' or c['midpoint_forwards']!=0
        or not c['original_inputs_unchanged'] or not c['derived_inputs_unchanged']
        or not c['actual_order_and_exposure_verified'] or c['TEST'] or s['TEST']
        or selection['representative_checkpoint']!=cp or selection['representative_epoch']!=3
        or selection['selected_epoch']!=3 or selection['gate_reference']!='R4 input B'
        or s['representative']['checkpoint']!=cp or len(s['epochs'])!=24
        or [e['epoch'] for e in s['epochs']]!=list(range(1,25))
        or training.io.pick(s['epochs'],p)['epoch']!=3):
        raise ValueError('Completed predeclared B-conditioned representative ep3 required')
    return cp


def identity():
    from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as previous
    c=checked_config();s=read(RUN/'core_summary.json');done=read(RUN/'completion.json')
    cp=validate_training(s,done,read(RUN/'selection.json'));protocol=read(RUN/'protocol.json')
    parent=p.checked_sources()
    if (s['source_SHA']!=training.sources() or protocol['identity']['sources']!=s['source_SHA']
        or protocol['identity']['source_cache_parent']!=parent):raise ValueError('Training source/parent drift')
    for n,h in read(RUN/'core_manifest.json').items():
        if p.sha(RUN/n)!=h:raise ValueError('Training core changed: '+n)
    for n,h in s['frozen_pins'].items():
        if p.sha(ROOT/n)!=h:raise ValueError('Frozen parent/cache changed: '+n)
    if p.sha(RUN/cp['path'])!=cp['sha256']:raise ValueError('LB weight SHA differs')
    if p.sha(previous.OUT/'test_rows.jsonl')!=L5_ROWS_SHA:raise ValueError('Sealed old TEST changed')
    verified=read(RUN/'terminal_verification.json')
    if not verified['ready_for_user_authorized_fixed_TEST']:raise ValueError('VAL terminal audit required')
    return dict(settings=c,checkpoint=cp,parent=parent,training_identity=protocol['identity'],
        sources={n:p.sha(ROOT/n) for n in EXTRA},frozen_pins=s['frozen_pins'],
        receipts={n:p.sha(RUN/n) for n in ('core_manifest.json','core_summary.json','completion.json',
            'selection.json','terminal_verification.json','epoch_03_VAL.rows.jsonl.gz')},
        old_TEST_rows_SHA=L5_ROWS_SHA,TEST_exposed=True,selection_on_test=False,training_updates=0,
        old_models_reinferred=False,midpoint_model_built=False,midpoint_forwards=0,efficiency=False)


def capture_B(detector,head,image,metas,counts):
    """GT-free online path. B supplies both the ROI and LVSR geometry condition."""
    import numpy as np
    import torch
    from crane_project.utils import port_geometry_refine_g_v1 as g
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    g.assert_detector_frozen(detector)
    if head.training or any(q.requires_grad for q in head.parameters()):raise ValueError('LB must be frozen')
    with torch.no_grad():
        features=detector.extract_feat(image);counts['features']+=1
        if features[0].shape!=(1,256,128,128) or any(f.requires_grad or f.grad_fn is not None for f in features):
            raise ValueError('Frozen P3 feature shape/graph differs')
        def native(rescale):
            value=detector.simple_test_from_features(features,metas,rescale=rescale);counts['native_heads']+=1
            if len(value)!=1 or len(value[0])!=1:raise ValueError('Single image/class required')
            a=np.asarray(value[0][0])
            if a.ndim!=2 or a.shape[1]!=6 or len(a)>1 or not np.isfinite(a).all():raise ValueError('Top1 native boxes differ')
            return a
        raw_np=native(False);original_np=native(True)
        raw,original=image.new_tensor(raw_np),image.new_tensor(original_np)
        restored=g.map_boxes(raw[:,:5],metas[0],inverse=True)
        if (raw.shape!=original.shape or not torch.allclose(restored,original[:,:5],atol=1e-4,rtol=1e-6)
            or not torch.equal(raw[:,5],original[:,5])):raise ValueError('B coordinate restoration differs')
        roi,support,_=g.sample_local(features[0],raw[:,:5],metas[0],'aligned')
        xy=original.new_tensor(np.asarray(metas[0]['scale_factor']).reshape(-1)[:2]).reshape(1,2).expand(len(original),-1)
        out=head(roi,support,original,raw[:,:5],xy,original);counts['LB_heads']+=1
        if not np.array_equal(raw_np,native(False)):raise ValueError('LB changed native B')
        if len(out['boxes_original'])!=len(original):raise ValueError('LB added/dropped output')
        present=bool(len(original));b=original[0].cpu().tolist() if present else None
        result=dict(b=b,midpoint=b,edge_residual=out['boxes_original'][0].cpu().tolist() if present else None,
            vector_valid=bool(out['vector_valid'][0]) if present else None,
            size_delivery='no_output' if not present else 'neutral' if bool(out['neutral_bypass'][0]) else
                'square' if bool(out['square_bypass'][0]) else 'accepted' if bool(out['accepted'][0]) else 'fallback',
            failed_checks=[k for k,v in out['checks'].items() if present and not bool(v[0])],
            geometry_condition='B',original_b_raw_exact_before_after=True)
        if not frozen_identity(result):raise ValueError('LB changed inherited B fields/axis order')
        return result


def merge_reference(current,sealed):
    from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as previous
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    if len(current)!=1440 or len(sealed)!=1440:raise ValueError('Full1440 paired TEST rows required')
    merged=[]
    for a,b in zip(current,sealed):
        if (any(a[k]!=b[k] for k in ('image','gt','domain','sequence','frame_id'))
            or a['midpoint']!=a['b'] or not frozen_identity(a)):raise ValueError('Frame/GT/B alias/inheritance differs')
        previous.compare_boxes(a,b,('b',))
        merged.append(dict(a,M20=b['midpoint'],LM=b['edge_residual']))
    return merged


def analyze(current,sealed):
    from crane_project.tools.eval_lvsr_partial_visual_norm_fixed_test_v1 import metric
    from crane_project.tools.eval_port_geometry_ap_v1 import verify_native
    merged=merge_reference(current,sealed);groups={'overall':merged}
    groups.update({d:[r for r in merged if r['domain']==d] for d in ('real','sim')})
    groups.update({s:[r for r in merged if r['sequence']==s] for s in sorted({r['sequence'] for r in merged})})
    metrics={};paired={};table=[];accurate=[];native={}
    for group,rr in groups.items():
        metrics[group]={};details={}
        for name,key in SPECS.items():
            m,v,ds,gs,ap=metric(rr,key);metrics[group][name]=m;details[name]=v;table.append(dict(group=group,method=name,**m))
            if group=='overall':
                for t in (.5,.75):native[name+'/'+str(t)]=verify_native(ds,gs,t,ap[str(t)])
        after=details['R4_LB'];paired[group]={}
        for name in ('R4','R4_M20','R4_M20_LM'):
            before=details[name]
            paired[group][name]=dict(fixed_joint=sum(not a['joint_size10'] and b['joint_size10'] for a,b in zip(before,after)),
                broken_joint=sum(a['joint_size10'] and not b['joint_size10'] for a,b in zip(before,after)),
                recovered75=sum(a['riou']<.75<=b['riou'] for a,b in zip(before,after)),
                lost75=sum(b['riou']<.75<=a['riou'] for a,b in zip(before,after)))
        for reference in ('R4','R4_M20'):
            for edge in ('long','short'):
                key=edge+'_relative_error';ii=[i for i,v in enumerate(details[reference]) if v['output'] and v[key]<=.03]
                e=dict(group=group,reference=reference,edge=edge,fixed_count=len(ii))
                for name in SPECS:
                    e[name+'_MAE_pct']=100*sum(details[name][i][key] for i in ii)/len(ii) if ii else None
                    e[name+'_harmed_vs_reference']=sum(details[name][i][key]>details[reference][i][key]+1e-6 for i in ii)
                accurate.append(e)
    return dict(metrics=metrics,paired=paired,native_AP=native,frames=1440,TEST_exposed=True,
        selection_on_test=False,condition_alias='midpoint is B internally; actual M20 is merged from sealed rows'),table,accurate


class Monitor:
    def __init__(self,c):self.c=c;self.done=threading.Event();self.error=None;self.samples=[]
    def sample(self):
        from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows,processes
        g=next(v for v in gpu_rows() if v['uuid']==self.c['gpu_uuid'])
        apps=[v for v in processes() if v['uuid']==self.c['gpu_uuid']]
        own=sum(v['used_mib'] for v in apps if v['pid']==os.getpid())
        if (any(v['pid']!=os.getpid() for v in apps) or own>=self.c['memory']['process_stop_mib']
            or g['free_mib']<self.c['memory']['minimum_free_mib']):raise RuntimeError('GPU3 contention/memory limit')
        r=dict(unix_time=time.time(),gpu=g,own_process_mib=own);self.samples.append(r)
        with (RUN/'test_LB_resources.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    def loop(self):
        while not self.done.is_set():
            try:self.sample()
            except BaseException as e:self.error=repr(e);return
            self.done.wait(2)
    def start(self):self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def check(self):
        if self.error:raise RuntimeError(self.error)
    def stop(self):self.done.set();self.thread.join(timeout=15);self.check()


def execute():
    for n in ('test_LB_lock.json','test_LB_rows.jsonl','test_LB_completion.json','test_LB_failure.json'):
        if (RUN/n).exists():raise ValueError('Preserve previous TEST, no automatic retry: '+n)
    locked=identity();c=locked['settings']
    save(RUN/'test_LB_lock.json',dict(identity=locked,user_authorized=True,efficiency_authorized=False))
    status('LB_TEST_startup');started=time.monotonic()
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    from crane_project.tools import eval_symeood_r4_midpoint_fixed_test_v1 as fixed
    from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as previous
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.tools.eval_port_geometry_ap_v1 import validate_saved_gt
    from PIL import Image
    import numpy as np
    adapted.resource_check(3,before=True)
    engine=adapted.make_engine(3);torch=engine.torch;runtime=engine.runtime(0);engine.seed_all();torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_optimizer=torch.optim.Optimizer.__init__
    def forbidden(*a,**k):raise RuntimeError('No optimizer allowed in frozen TEST')
    torch.optim.Optimizer.__init__=forbidden
    monitor=Monitor(c);monitor.start()
    try:
        cfg=engine.check_cfg();detector=engine.build_detector(cfg,0)
        if runtime!=read(RUN/'protocol.json')['runtime']:raise ValueError('Training runtime differs')
        payload=torch.load(str(RUN/locked['checkpoint']['path']),map_location='cpu')
        if (payload['identity']!=locked['training_identity'] or payload['condition']!='B' or payload['epoch']!=3
            or payload['updates']!=1917 or payload['parameters']!=2480 or payload['midpoint_model_used']
            or payload['detector_sha256']!=p.parent.R4_SHA):raise ValueError('LB checkpoint payload differs')
        head=PairSpecificSizeHead().cuda();head.load_state_dict(payload['head_state'],strict=True);del payload
        for m in (detector,head):m.eval().requires_grad_(False)
        from crane_project.tools.smoke_faa_gpu_v1 import processes
        if {a['uuid'] for a in processes() if a['pid']==os.getpid()}!={c['gpu_uuid']}:raise ValueError('Physical GPU mapping differs')
        def states():return {n:engine.g.state_digest(m) for n,m in [('R4',detector),('LB',head)]}
        before=states();counts=dict(features=0,native_heads=0,LB_heads=0)
        status('LB_TEST_VAL_replay')
        with gzip.open(RUN/'epoch_03_VAL.rows.jsonl.gz','rt') as f:reference={r['image']:r for r in map(json.loads,f)}
        data=engine.g.ready.DATA;manifest=read(data/'manifest.json');metadata={r['id']:r for r in manifest['records'] if r['split']=='val'}
        dataset=engine.dataset_for(cfg,'val',1.);lookup={Path(v['filename']).stem:i for i,v in enumerate(engine.infos(dataset))};proof=[]
        with torch.no_grad():
            for name in fixed.VAL_NAMES:
                ref=reference[name];meta=metadata[name]
                for k in ('images','annfiles'):
                    if p.sha(data/meta[k]['path'])!=meta[k]['sha256']:raise ValueError('VAL bytes differ')
                with Image.open(data/meta['images']['path']) as im:source=dict(ref,image_size=list(im.size))
                image,metas=engine.old.load_view(dataset,lookup[name],source,0);actual=capture_B(detector,head,image,metas,counts)
                error=previous.compare_boxes(dict(actual,image=name),ref,('b','midpoint','edge_residual'))
                previous.compare_decisions(actual,ref);proof.append(dict(image=name,max_abs=error))
        save(RUN/'test_LB_VAL_replay.json',dict(passed=True,frames=8,proof=proof,midpoint_forwards=0))
        status('LB_TEST_input_identity');sources,data_identity=engine.old.fixed_test_inputs()
        if p.sha(ROOT/fixed.TEST_REFERENCE)!=fixed.TEST_REFERENCE_SHA:raise ValueError('Sealed GT reference changed')
        old=rows(previous.OUT/'test_rows.jsonl');sealed=rows(ROOT/fixed.TEST_REFERENCE)
        if len(sources)!=1440 or len(old)!=1440 or len(sealed)!=1440:raise ValueError('1440 TEST frames required')
        for source,ref,v in zip(sources,sealed,old):
            if source['image']!=ref['image'] or source['image']!=v['image'] or ref['gt']!=v['gt']:raise ValueError('TEST identity differs')
            poly=np.asarray([float(x) for x in (data/'test/annfiles'/(source['image']+'.txt')).read_text().split()[:8]]).reshape(4,2)
            validate_saved_gt(ref['gt'],poly,np.asarray(source['gt']));source['gt']=ref['gt']
        dataset=engine.old.build_test_dataset(cfg,sources);current=[];maximum=0.
        status('LB_TEST',done=0,total=1440)
        with torch.no_grad(),(RUN/'test_LB_rows.jsonl').open('x') as f:
            for i,source in enumerate(sources):
                image,metas=engine.old.load_view(dataset,i,source,0);value=capture_B(detector,head,image,metas,counts)
                row=dict(image=source['image'],sequence=source['sequence'],domain=source['domain'],frame_id=source['frame_id'],
                    gt=source['gt'],scale=1.,split='test',**value)
                maximum=max(maximum,previous.compare_boxes(row,old[i],('b',))['b'])
                current.append(row);f.write(json.dumps(row,allow_nan=False)+'\n')
                if i%50==0:monitor.check();engine.resource_check(0);status('LB_TEST',done=i+1,total=1440);f.flush()
        if counts!=dict(features=1448,native_heads=4344,LB_heads=1448):raise ValueError('Inference counts differ')
        if states()!=before or identity()!=locked:raise ValueError('Frozen state/source drift')
        status('LB_TEST_analysis',done=1440,total=1440);report,table,accurate=analyze(current,old)
        report.update(protocol=c['protocol'],checkpoint=locked['checkpoint'],training_updates=0,automatic_adoption=False,
            delivery_counts=dict(Counter(r['size_delivery'] for r in current)),efficiency=False)
        save(RUN/'test_LB_summary.json',report)
        for name,rr in [('test_LB_metrics.csv',table),('test_LB_accurate_edges.csv',accurate)]:
            with (RUN/name).open('x',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
        monitor.stop()
        if states()!=before or identity()!=locked:raise ValueError('Final state/source drift')
        save(RUN/'test_LB_completion.json',dict(passed=True,frames=1440,checkpoint=locked['checkpoint'],state_before=before,
            state_after=states(),audit_counts=counts,baseline_B_max_abs=maximum,data_identity=data_identity,runtime=runtime,
            training_updates=0,selection_on_test=False,TEST_exposed=True,midpoint_model_built=False,midpoint_forwards=0,
            rows_SHA=p.sha(RUN/'test_LB_rows.jsonl'),driver_samples=len(monitor.samples),
            driver_peak_mib=max(v['own_process_mib'] for v in monitor.samples),
            minimum_free_mib=min(v['gpu']['free_mib'] for v in monitor.samples),elapsed_seconds=time.monotonic()-started,
            efficiency=False,old_models_reinferred=False))
        names=['test_LB_lock.json','test_LB_VAL_replay.json','test_LB_summary.json','test_LB_metrics.csv',
            'test_LB_accurate_edges.csv','test_LB_completion.json']
        save(RUN/'test_LB_core_manifest.json',{n:p.sha(RUN/n) for n in names})
        status('LB_TEST_COMPLETE',epoch=3,frames=1440)
    except BaseException as e:
        save(RUN/'test_LB_failure.json',dict(error=repr(e),TEST=True,training_updates=0,retry=False))
        status('LB_TEST_FAILED',error=repr(e));raise
    finally:
        monitor.done.set();torch.optim.Optimizer.__init__=original_optimizer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:print(json.dumps(checked_config(),indent=2));return
    def timeout(*unused):raise RuntimeError('Fixed3600-second TEST budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,checked_config()['timeout_seconds'])
    try:execute()
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
