#!/usr/bin/env python3
"""Bounded, read-only TRAIN supervision/pixel evidence check. CPU only."""
import argparse
from copy import deepcopy
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import preflight_port_geometry_g_center_only_v1 as o
from crane_project.utils import port_train_geometry_evidence_v1 as e

VERSION='port_train_geometry_evidence_v1'
PROTOCOL=ROOT/'crane_project/tools/port_train_geometry_evidence_v1_protocol.json'
MANIFEST=ROOT/'crane_project/tools/port_train_geometry_evidence_v1_sources.json'
PRIOR_MANIFEST_SHA='8049731476d380866e8969448c0fb54d18787340e542ff793bab34e5b72f9017'
CENTER_REPORT_SHA='5f7c70f1f3e65147d9d568fbbe9da23591bf7bd2d83ef68a6b59ad5bdda6b081'
BASELINE=ROOT/'work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json'
JOINT=ROOT/'work_dirs/port_geometry_g_center_v1_train/completion.json'
CENTER=ROOT/'work_dirs/port_geometry_g_center_only_v1_train/completion.json'
ARMS=('b','ordinary','aligned','joint','joint_center_b_shape','center_only')


def protocol_document():
    return dict(protocol=VERSION,evidence_role='bounded_repeated_TRAIN_geometry_and_pixel_diagnosis',
        settings=deepcopy(e.SETTINGS),arms=list(ARMS),
        baseline_report_sha256=o.c.a.BASELINE_SHA,joint_report_sha256=o.JOINT_REPORT_SHA,
        center_only_report_sha256=CENTER_REPORT_SHA,prior_manifest_sha256=PRIOR_MANIFEST_SHA,
        scope=dict(train_images=64,related_views=128,image_render_limit=12,
            detector_forward_calls=0,detector_updates=0,head_updates=0,
            feature_cache_loads=0,feature_measurements=0,gpu_devices_used=[],
            val_access=False,test_access=False,checkpoint_export=False,label_changes=False),
        grouping='Every group keeps fit/probe, real/sim and scale1/.5 separate. Further stratify by sequence, frozen B center px, GT input short-edge stride8 units, GT aspect. No pooling related views for independent-sample claims.',
        selection='Within each fit/probe x real/sim stratum, in order: center-worse AND RIoU-worse sorted by greatest RIoU loss, greater center loss, image, scale; then center-better AND RIoU-better sorted by greatest RIoU gain, greater center gain, image, scale; then scale1 view closest to frozen B scale1 center-error median, image tie break. Exclude already selected image including its other scale. Empty slot stays unavailable, no fallback. Display both scales. Tolerances apply only to sign classification.',
        metrics='Recompute strict rotated IoU and original-coordinate geometry from saved raw boxes, retaining raw w/h/angle/score association. Center<15px only output frames; also output coverage and all-frame correct, all fractions explicit. Pure periodic angle and 10px protocol angle penalty distinct. Center-only proxy loss uses detached original B short edge and original (px+py)/3.',
        supervision='Native CraneDataset poly2obb_np le90; compare raw DOTA quadrilateral, fitted rectangle, reference GT, corner residual, image-edge margin. Record native parse differences; reference is NEVER replaced. Selected image and annotation bytes must match fixture SHA. No automatic semantic-label/visibility verdict.',
        pixel_views='Native LoadImageFromFile decoded BGR pixels, channel permutation to RGB; native RResize1024 keep_ratio and fixed PortIsotropicShrink.5, no flip. Preserve dimension rounding, GT sizes*sqrt(sx*sy), detector raw w/h*sx/sy. Display-only crop/enlargement uses same uniform affine for pixels and overlays. Separate raw and overlay tiles; no sharpening. Stride8 dots are coordinate convention only, NOT measured FPN activations or precision limit.',
        static='Verify source, TRAIN fixture, selected image/annotation bytes and reviewed report identities; recompute128 saved views and fixed selection. No image pixel rendering, heads, ROI cache, GPU or updates.',
        review='Full run completes with HUMAN_REVIEW_REQUIRED, never approves training or assigns annotation faults. Fill worksheet after checking object scope, boundary visibility, occlusion and visible support for desired versus actual correction. Only if these first two checks remain inconclusive consider a separate frozen ROI/P3 spatial evidence check.',
        limitations=['TRAIN probe16 identities/32 correlated views is repeatedly development-used; B trained on all64. Not independent VAL.',
            'B epoch24 remains retained; no new checkpoint/architecture/loss/threshold choice.',
            'Diagnostic extremes are not representative prevalence; all128 views are reported.',
            'No direct FPN or ROI-response measurement; valid-mask support is not feature quality.',
            'Images can expose inconsistencies but cannot prove true subpixel target geometry or unique failure cause.',
            'TEST repeatedly exposed; no TEST access/tuning/selection. Sparse TRAIN coverage does not establish video continuity or depth precision.'])


def checked_sources():
    manifest=json.loads(MANIFEST.read_text()); protocol=json.loads(PROTOCOL.read_text())
    if (manifest.get('protocol')!=VERSION or manifest.get('settings')!=e.SETTINGS
            or protocol!=protocol_document()):
        raise ValueError('Predeclared evidence protocol/settings differs')
    if o.g.ready.sha(o.MANIFEST)!=PRIOR_MANIFEST_SHA:
        raise ValueError('Reviewed center-only provenance changed')
    expected=set(json.loads(o.MANIFEST.read_text())['sources']) | {
        'crane_project/tools/check_port_train_geometry_evidence_v1.py',
        'crane_project/utils/port_train_geometry_evidence_v1.py',
        'crane_project/tools/port_train_geometry_evidence_v1_protocol.json'}
    if set(manifest['sources'])!=expected:
        raise ValueError('Incomplete evidence source set')
    actual={p:o.g.ready.sha(ROOT/p) for p in manifest['sources']}
    if actual!=manifest['sources']:
        raise ValueError('Evidence source SHA differs: '+', '.join(p for p in actual if actual[p]!=manifest['sources'][p]))
    return dict(manifest_sha256=o.g.ready.sha(MANIFEST),protocol_sha256=o.g.ready.sha(PROTOCOL),sources=actual)


def verify_metrics(rows):
    """Independent saved-box calculation; tolerate declared OpenCV rounding."""
    maxima=dict(riou=0.,other_geometry=0.)
    for row in rows:
        p=row['pred']; calc=o.g.decompose(row['gt_original'],p[:5] if p is not None else None)
        if set(calc)!=set(row['metrics']): raise ValueError('Saved metric fields differ')
        for key,value in calc.items():
            saved=row['metrics'][key]
            if value is None or isinstance(value,(str,bool)):
                if saved!=value: raise ValueError('Saved metric flag differs: '+key)
            else:
                if saved is None or not math.isfinite(float(saved)): raise ValueError('Nonfinite saved metric')
                delta=abs(float(saved)-float(value)); is_riou=key=='riou' or key.startswith('riou_gain_')
                maxima['riou' if is_riou else 'other_geometry']=max(maxima['riou' if is_riou else 'other_geometry'],delta)
                if delta>(1e-5 if is_riou else 1e-8): raise ValueError('Saved box geometry mismatch: '+key)
    return maxima


def checked_inputs(baseline_path,joint_path,center_path):
    identity=checked_sources()
    _,samples,prior,baseline,joint,hybrid=o.checked_contract(baseline_path,joint_path)
    if o.g.ready.sha(center_path)!=CENTER_REPORT_SHA:
        raise ValueError('Requires exact reviewed center-only completion')
    center=json.loads(center_path.read_text())
    if (center['protocol']!=o.VERSION or center['identity']!=prior
            or center['status']!='TRAIN_CENTER_ONLY_CHECK_COMPLETE_REVIEW_REQUIRED'
            or center['head_updates_total']!=200 or center['additional_joint_updates']!=0
            or center['detector_forward_calls']!=0 or center['detector_updates']!=0
            or center['formal_training'] or center['checkpoint_exported']
            or not center['detector_state_unchanged'] or center['minibatches']!=baseline['minibatches']):
        raise ValueError('Center-only reviewed identity/scope differs')
    o.verify_initial(center['center_only']['initial'],baseline,joint)
    arms=dict(b=baseline['arms']['ordinary']['initial'],
        ordinary=baseline['arms']['ordinary']['final'],aligned=baseline['arms']['aligned']['final'],
        joint=joint['joint_center']['final'],joint_center_b_shape=hybrid,center_only=center['center_only']['final'])
    recompute={}
    for label,arm in arms.items():
        o.c.paired_changes(arms['b'],arm)
        if len(arm['rows'])!=128 or not all(r['eligible'] and r['pred'] is not None for r in arm['rows']):
            raise ValueError('Expected reviewed128 eligible outputs, no filtering')
        recompute[label]=verify_metrics(arm['rows'])
    for br,cr in zip(arms['b']['rows'],arms['center_only']['rows']):
        if br['pred'][2:]!=cr['pred'][2:]: raise ValueError('Center-only raw shape/score changed')
    annotations={}; evidence={}
    for sample in samples:
        ann=e.read_annotation(o.g.ready.DATA/sample['split']/'annfiles'/(sample['image']+'.txt'))
        annotations[sample['image']]=ann; evidence[sample['image']]=e.annotation_evidence(sample,ann)
    rows=e.build_rows(arms,samples,evidence); groups=e.grouped_report(rows)
    loss_delta=0.; cells_delta=0.
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for scale in (1.,.5):
                key=role+'/'+domain+'/'+str(scale); group=groups[key+'/all']
                d=abs(group['arms']['center_only']['center_loss']['mean']-center['center_only']['final']['eligible_center_loss'][key])
                loss_delta=max(loss_delta,d)
                a=group['gt_short_cells']; b=baseline['train_support'][key]['gt_short_cells']
                cells_delta=max(cells_delta,max(abs(a[k]-b[k]) for k in ('mean','median','p90','rmse')))
    if loss_delta>1e-6 or cells_delta>1e-5: raise ValueError('Center loss or native GT input geometry differs')
    identity.update(prior_identity=prior,baseline_report_sha256=o.g.ready.sha(baseline_path),
        joint_report_sha256=o.g.ready.sha(joint_path),center_only_report_sha256=o.g.ready.sha(center_path))
    analysis=dict(protocol=VERSION,rows=rows,groups=groups,annotation_rows=list(evidence.values()),
        native_parse_reference_review_images=[k for k,v in evidence.items() if v['native_parse_reference_status']=='NATIVE_PARSE_DIFFERENCE_REVIEW_REQUIRED'],
        recompute=dict(saved_boxes=recompute,center_loss_max_absolute_delta=loss_delta,
            native_gt_short_cells_max_absolute_delta=cells_delta),
        reused_spatial_support=deepcopy(baseline['train_support']),
        spatial_support_note='Original saved valid-mask / size support only; no actual feature response measured.',
        limitations=protocol_document()['limitations'])
    return samples,annotations,identity,analysis,e.select_images(rows)


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temporary.replace(path)


def write_human_review(out,selection,panels):
    fields=['image','role','domain','category','trigger_scale','panel','object_scope_consistent',
        'visible_boundary_consistent','occlusion_or_ambiguity','desired_center_shift_visible',
        'native_scale1_boundary_support','native_scale0_5_boundary_support','reviewer','notes']
    reviews=[]
    for slot in selection['slots']:
        if slot['status']!='SELECTED': continue
        row={k:slot[k] for k in fields[:5]}; row['panel']='panels/'+slot['image']+'.png'
        row.update({k:'NOT_REVIEWED' for k in fields[6:-2]}); row.update(reviewer='',notes='')
        reviews.append(row)
    with (out/'human_review_template.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(reviews)
    body=['<!doctype html><meta charset="utf-8"><title>TRAIN geometry evidence</title>',
        '<style>body{font:16px sans-serif;max-width:1320px;margin:24px auto}img{width:100%}code{word-wrap:break-word}</style>',
        '<h1>TRAIN 几何监督与像素证据核对</h1><p>只读、零更新；保留 B epoch24。',
        '以下为结果引导的诊断例，不能估计总体比例或视为独立 VAL。请结合 analysis.json 中全部128视图及 human_review_template.csv。</p>',
        '<p>人工核对：标注是否包围同一对象范围；边界与遮挡是否使中心/方向不明确；所需纠偏是否有可见依据；',
        '原尺度及半尺度输入是否仍有可辨边界。可填写 YES/NO/UNCERTAIN 并说明依据，不因模型退化而修改标签。',
        '原始图与框分开显示；网格点仅为 stride8 坐标，不是特征响应。</p>',
        '<p>保持原始模板及 SHA 清单不变；如填写人工判断，请另存 human_review_completed.csv。',
        '该后续人工文件不在运行完成时的 SHA 清单内，也不会自动改为审查通过。</p>']
    for slot,panel in zip([s for s in selection['slots'] if s['status']=='SELECTED'],panels):
        label=' / '.join(str(slot[k]) for k in ('image','role','domain','category','trigger_scale'))
        body.append('<h2>'+html.escape(label)+'</h2><img src="panels/'+html.escape(panel['file'],quote=True)+'">')
    body.append('<p>尚未人工审查的字段全部 NOT_REVIEWED；本报告不自动批准训练或证明根因。</p>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf-8')


def run(args):
    out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    progress_path=out/'progress.jsonl'; start=time.monotonic()
    scope=dict(detector_forward_calls=0,detector_updates=0,head_updates=0,feature_cache_loads=0,
        feature_measurements=0,gpu_devices_used=[],val_access=False,test_access=False,
        formal_training=False,checkpoint_exported=False,annotation_changes=False)
    def progress(value):
        with progress_path.open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
    completion=dict(protocol=VERSION,scope=scope,check_only=args.check_only,
        status='RUNNING',rendered_images=0,formal_training_approved=False,human_review='NOT_REVIEWED')
    try:
        progress(dict(stage='begin',check_only=args.check_only))
        samples,annotations,identity,analysis,selection=checked_inputs(
            Path(args.baseline_report),Path(args.joint_report),Path(args.center_only_report))
        completion.update(identity=identity,verified_images=len(samples),verified_views=len(analysis['rows']),
            selected_images=selection['unique_images'],recompute=analysis['recompute'],
            native_parse_reference_review_images=analysis['native_parse_reference_review_images'],
            runtime=dict(python=sys.version,numpy=np.__version__,opencv=e.cv2.__version__,
                opencv_build_sha256=hashlib.sha256(e.cv2.getBuildInformation().encode()).hexdigest()))
        write_json(out/'protocol.json',protocol_document()); write_json(out/'selection.json',selection)
        write_json(out/'analysis.json',analysis)
        progress(dict(stage='verified',images=len(samples),views=len(analysis['rows'])))
        panels=[]
        if not args.check_only:
            (out/'panels').mkdir(); lookup={s['image']:s for s in samples}
            for slot in selection['slots']:
                if slot['status']!='SELECTED': continue
                sample=deepcopy(lookup[slot['image']]); sample['path']=o.g.ready.DATA/sample['split']/'images'/(sample['image']+'.jpg')
                # Revalidate immediately before rendering. Never follow an edited image silently.
                annotation_path=o.g.ready.DATA/sample['split']/'annfiles'/(sample['image']+'.txt')
                if (o.g.ready.sha(sample['path'])!=sample['image_sha256']
                        or o.g.ready.sha(annotation_path)!=sample['annotation_sha256']):
                    raise ValueError('Selected image/annotation changed before render')
                rows=[r for r in analysis['rows'] if r['image']==sample['image']]
                panel=e.render_panel(sample,annotations[sample['image']],rows,out/'panels'/(sample['image']+'.png'))
                panels.append(panel); completion['rendered_images']=len(panels)
                progress(dict(stage='panel',image=sample['image'],completed=len(panels)))
                print('TRAIN evidence panel',len(panels),'/',selection['unique_images'],sample['image'],flush=True)
            write_json(out/'panel_manifest.json',dict(panels=panels,overlay_arm='full_joint_with_its_shape',
                extra_analysis_arm='saved_joint_center_b_shape isolates center; never separately trained'))
            write_human_review(out,selection,panels)
        completion['status']=('STATIC_TRAIN_GEOMETRY_EVIDENCE_COMPLETE_NO_PIXEL_RENDER_NO_UPDATES' if args.check_only
            else 'TRAIN_GEOMETRY_EVIDENCE_COMPLETE_HUMAN_REVIEW_REQUIRED')
        progress(dict(stage='complete',status=completion['status']))
    except Exception as error:
        completion.update(status='FAILED',error=str(error),error_type=type(error).__name__)
        progress(dict(stage='failed',error=str(error)))
        raise
    finally:
        completion['elapsed_seconds']=time.monotonic()-start; write_json(out/'completion.json',completion)
        files={str(p.relative_to(out)):o.g.ready.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'}
        write_json(out/'artifacts.json',dict(protocol=VERSION,status=completion['status'],files=files,
            excluded=['model_weights','ROI_or_FPN_payload','source_code_bundle']))
    print('Saved',out/'completion.json','status',completion['status'])
    return completion


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--baseline-report',default=str(BASELINE))
    p.add_argument('--joint-report',default=str(JOINT))
    p.add_argument('--center-only-report',default=str(CENTER))
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':
    print('TRAIN64 images /128 saved views, <=12 pixel panels. CPU only; no models, cache loads or updates.',flush=True)
    run(parser().parse_args())
