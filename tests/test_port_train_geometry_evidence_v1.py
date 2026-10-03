"""CPU regressions for original-coordinate evidence, scope and audit outputs."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from crane_project.tools import check_port_train_geometry_evidence_v1 as q
e=q.e


def fixture_rows():
    samples=[]; rows=[]
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for index,error in enumerate((3.,2.,1.)):
                image=domain+'_seq01_'+role+str(index)
                samples.append(dict(image=image,role=role,domain=domain,sequence=domain+'_seq01',
                    image_size=[1031,721],gt=[100.,100.,40.,20.,.2]))
                for scale in (1.,.5):
                    p=[100.+error,100.,40.,20.,.2,.8]
                    rows.append(dict(image=image,role=role,domain=domain,sequence=domain+'_seq01',frame_id=index,
                        scale=scale,eligible=True,gt_original=[100.,100.,40.,20.,.2],pred=p,metrics=q.o.g.decompose([100.,100.,40.,20.,.2],p[:5])))
    changed=deepcopy(rows)
    for row in changed:
        index=row['frame_id']; row['pred'][0]+=(1. if index==0 else -1. if index==1 else 0.)
        row['metrics']=q.o.g.decompose(row['gt_original'],row['pred'][:5])
    annotations={s['image']:{} for s in samples}
    return samples,dict(b=dict(rows=rows),center_only=dict(rows=changed)),annotations


def test_raw_width_height_angle_association_and_physical_polygon():
    b=[40.,60.,20.,50.,.3,.8]
    p=e.box_polygon(b)
    assert np.allclose(p.mean(axis=0),b[:2])
    assert np.linalg.norm(p[1]-p[0])==pytest.approx(20.)
    assert np.linalg.norm(p[2]-p[1])==pytest.approx(50.)
    equiv=b.copy(); equiv[2:4]=[50.,20.]; equiv[4]+=math.pi/2
    assert np.allclose(sorted(map(tuple,np.round(p,8))),sorted(map(tuple,np.round(e.box_polygon(equiv),8))))
    geometry=e.view_geometry([1031,721],.5); pb=e.model_box(b,geometry)
    assert pb[2]==pytest.approx(20.*geometry['scale_factor'][0])
    assert pb[3]==pytest.approx(50.*geometry['scale_factor'][1])
    assert pb[4:]==pytest.approx(b[4:])
    with pytest.raises(ValueError): e.box_polygon([1.,2.,0.,3.,0.])


@pytest.mark.parametrize('scale',[1.,.5])
def test_odd_size_native_pixels_rounding_and_annotation_geometry(scale,tmp_path):
    original=Image.new('RGB',(1031,721),(10,20,30)); path=tmp_path/'pixels.png'; original.save(path)
    rgb=e.native_rgb(path)
    assert np.array_equal(np.asarray(rgb),np.asarray(original))
    view,g=e.native_view(rgb,scale)
    assert g['resize_wh']==[1024,716]
    assert list(view.size)==[math.ceil(1024*scale),math.ceil(716*scale)]
    sx,sy=g['scale_factor'][:2]
    gt=e.model_box([100.,200.,40.,20.,.2],g,annotation=True)
    assert gt[:2]==pytest.approx([100.*sx,200.*sy])
    assert gt[2:4]==pytest.approx(np.array([40.,20.])*math.sqrt(sx*sy))
    assert g['flip'] is False and g['pad_wh']==[1024,1024]
    with pytest.raises(ValueError,match='Undeclared'): e.view_geometry([1031,721],.75)


def test_center_proxy_has_original_two_one_third_weights_and_frozen_b_short():
    b=[30.,40.,40.,20.,.2,.8]; gt=[31.,38.,400.,2.,-1.]
    part=e.center_parts(b,gt,b)
    err=np.array([-1.,2.])/6.
    expected=(abs(err)-.05).sum()/3.
    assert part['weighted_sum']==pytest.approx(expected)
    changed=b.copy(); changed[2:4]=[400.,200.]
    assert e.center_parts(changed,gt,b)==part
    assert e.center_parts(None,gt,b) is None


def test_signed_shifts_stratification_preserves_roles_scales_and_raw_reference():
    samples,arms,annotations=fixture_rows(); original=deepcopy(arms)
    rows=e.build_rows(arms,samples,annotations); groups=e.grouped_report(rows)
    assert len(rows)==24 and arms==original
    assert len([k for k in groups if k.endswith('/all')])==8
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for scale in (1.,.5):
                group=groups[role+'/'+domain+'/'+str(scale)+'/all']
                assert group['views']==3
                assert all(image.startswith(domain) and role in image for image in group['image_names'])
                assert group['arms']['b']['center_px']['mean']==pytest.approx(2.)
                assert group['arms']['center_only']['center_better']==1
                assert group['arms']['center_only']['center_worse']==1
    row=rows[0]
    assert row['desired_shift_px']==[-3.,0.]
    assert row['arms']['center_only']['actual_shift_px']==[1.,0.]
    assert row['arms']['center_only']['signed_residual_px']==[4.,0.]
    assert row['arms']['center_only']['direction_cosine']==-1.
    assert row['desired_shift_over_b_short']==[-.15,0.]
    group=groups['fit/real/1.0/all']
    assert group['desired_shift_sign_counts']['x']==dict(negative=3,positive=0,zero=0)
    assert group['arms']['center_only']['actual_shift_sign_counts']['x']==dict(negative=1,positive=1,zero=1)
    assert group['arms']['center_only']['signed_residual_over_b_short']['x']['mean']==pytest.approx(.1)
    assert 'not measured FPN' in row['phase_note']


def test_denominators_distinguish_missing_output_and_conditional_miss():
    samples,arms,annotations=fixture_rows()
    subset=[r for r in arms['b']['rows'] if r['role']=='fit' and r['domain']=='real' and r['scale']==1.]
    subset[0]['pred']=None; subset[0]['metrics']=q.o.g.decompose(subset[0]['gt_original'],None)
    subset[1]['pred'][0]=116.; subset[1]['metrics']=q.o.g.decompose(subset[1]['gt_original'],subset[1]['pred'][:5])
    groups=e.grouped_report(e.build_rows(dict(b=dict(rows=subset),center_only=dict(rows=deepcopy(subset))),samples,annotations))
    m=groups['fit/real/1.0/all']['arms']['b']
    assert m['output_coverage']==dict(numerator=2,denominator=3,pct=100.*2/3)
    assert m['conditional_center_correct']==dict(numerator=1,denominator=2,pct=50.)
    assert m['all_frame_center_correct']==dict(numerator=1,denominator=3,pct=100./3)
    assert m['riou']['n']==3 and m['center_px']['n']==2


def test_fixed_selection_unique_both_scales_median_control_and_no_fallback():
    samples,arms,annotations=fixture_rows(); rows=e.build_rows(arms,samples,annotations)
    selected=e.select_images(rows)
    assert selected==e.select_images(list(reversed(rows)))
    assert selected['unique_images']==12
    for slot in selected['slots']:
        assert slot['display_scales']==[1.,.5]
        expected_index={'negative':0,'positive':1,'ordinary':2}[slot['category']]
        assert slot['image'].endswith(str(expected_index))
    # No joint regression means empty positive/negative slots; don't fill them
    # with another scale of ordinary or quietly expand the diagnostic budget.
    for r in rows: r['arms']['center_only']=deepcopy(r['arms']['b'])
    none=e.select_images(rows)
    assert none['unique_images']==4
    assert sum(s['status']=='UNAVAILABLE_NO_FALLBACK' for s in none['slots'])==8


def test_annotation_geometry_not_semantic_verdict_and_native_parse_not_relabeling(tmp_path):
    box=[100.,100.,40.,20.,.2]; poly=e.box_polygon(box)
    p=tmp_path/'ann.txt'; p.write_text(' '.join(map(str,poly.reshape(-1)))+' grab 0\n')
    ann=e.read_annotation(p); sample=dict(image='real_seq01_00000',image_size=[200,200],gt=box)
    ev=e.annotation_evidence(sample,ann)
    assert ev['semantic_label_correctness']=='NOT_REVIEWED' and ev['visibility']=='NOT_REVIEWED'
    assert ev['polygon_rectangle_area_ratio']==pytest.approx(1.,abs=1e-5)
    assert ev['polygon_to_fitted_rectangle_corner_max_px']<1e-3
    sample['gt'][0]+=.03; before=deepcopy(sample)
    different=e.annotation_evidence(sample,ann)
    assert different['native_parse_reference_status']=='NATIVE_PARSE_DIFFERENCE_REVIEW_REQUIRED'
    assert sample==before and different['saved_reference_obb'][:2]==sample['gt'][:2]
    p.write_text(' '.join(map(str,poly.reshape(-1)))+' unknown 0\n')
    with pytest.raises(ValueError,match='strict TRAIN'): e.read_annotation(p)


def test_display_pixels_and_overlay_share_one_uniform_affine():
    image=np.zeros((30,40,3),dtype=np.uint8); image[9,12]=[255,0,0]
    bounds=[7,4,36,27]; tile,t=e.crop_tile(Image.fromarray(image),bounds)
    xy=e.display_points([[12,9]],t)[0]; x,y=np.round(xy).astype(int)
    assert np.array_equal(np.asarray(tile)[y,x],[255,0,0])
    assert t['uniform_scale'] and t['display_zoom']<=6.
    displacement=e.display_points([[13,9]],t)[0]-xy
    assert displacement==pytest.approx([t['display_zoom'],0.])
    assert e.crop_bounds([np.array([[0.,0.],[20.,20.]])],[40,30])[0:2]==[0,0]


def test_metric_recompute_rejects_changed_flag_or_nonfinite_and_preserves_data():
    _,arms,_=fixture_rows(); rows=arms['b']['rows']; before=deepcopy(rows)
    assert q.verify_metrics(rows)==dict(riou=0.,other_geometry=0.)
    assert rows==before
    rows[0]['metrics']['center_hit']=False
    with pytest.raises(ValueError,match='flag'): q.verify_metrics(rows)
    rows[0]=deepcopy(before[0]); rows[0]['metrics']['riou']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'): q.verify_metrics(rows)


def test_source_checks_reject_own_or_prior_drift_before_data(monkeypatch,tmp_path):
    assert len(q.checked_sources()['sources'])==65
    m=json.loads(q.MANIFEST.read_text()); m['sources'][next(iter(m['sources']))]='bad'
    manifest=tmp_path/'sources.json'; manifest.write_text(json.dumps(m))
    monkeypatch.setattr(q,'MANIFEST',manifest)
    with pytest.raises(ValueError,match='source SHA'): q.checked_sources()
    real_sha=q.o.g.ready.sha
    monkeypatch.setattr(q.o.g.ready,'sha',lambda p:'bad' if Path(p)==q.o.MANIFEST else real_sha(p))
    with pytest.raises(ValueError,match='provenance changed'): q.checked_sources()


def mock_input(tmp_path):
    samples,arms,annotations=fixture_rows(); arms['joint']=deepcopy(arms['center_only'])
    rows=e.build_rows(arms,samples,annotations)
    sample=samples[0]; sample['split']='train'; sample['path']=tmp_path/'train/images'/(sample['image']+'.jpg')
    sample['path'].parent.mkdir(parents=True); Image.new('RGB',tuple(sample['image_size']),(20,40,60)).save(sample['path'])
    sample['image_sha256']=q.o.g.ready.sha(sample['path'])
    ann=dict(polygon=e.box_polygon(sample['gt']).tolist())
    ap=tmp_path/'train/annfiles'/(sample['image']+'.txt'); ap.parent.mkdir(parents=True)
    ap.write_text(' '.join(map(str,e.box_polygon(sample['gt']).reshape(-1)))+' grab 0\n')
    sample['annotation_sha256']=q.o.g.ready.sha(ap)
    selected=dict(unique_images=1,slots=[dict(status='SELECTED',image=sample['image'],role='fit',domain='real',
        category='ordinary',trigger_scale=1.,display_scales=[1.,.5])])
    analysis=dict(rows=[r for r in rows if r['image']==sample['image']],groups={},recompute={},native_parse_reference_review_images=[])
    return [sample],{sample['image']:ann},{},analysis,selected


def test_static_and_full_entry_never_initialize_gpu_head_or_cache_and_artifacts_match(monkeypatch,tmp_path):
    data=mock_input(tmp_path); monkeypatch.setattr(q,'checked_inputs',lambda *_:deepcopy(data))
    monkeypatch.setattr(q.o.g.ready,'DATA',tmp_path)
    def forbidden(*_args,**_kwargs): raise AssertionError('GPU/head/cache execution prohibited')
    monkeypatch.setattr(torch.cuda,'init',forbidden); monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(q.o,'CenterOnlyLocalGeometryRefiner',forbidden)
    real_renderer=e.render_panel; monkeypatch.setattr(e,'render_panel',forbidden)
    static=q.run(q.parser().parse_args(['--check-only','--out-dir',str(tmp_path/'static')]))
    assert static['rendered_images']==0 and static['scope']['gpu_devices_used']==[]
    assert not (tmp_path/'static/panels').exists()
    monkeypatch.setattr(e,'render_panel',real_renderer)
    full=q.run(q.parser().parse_args(['--out-dir',str(tmp_path/'full')]))
    assert full['rendered_images']==1 and full['status'].endswith('HUMAN_REVIEW_REQUIRED')
    assert full['formal_training_approved'] is False and full['human_review']=='NOT_REVIEWED'
    files=json.loads((tmp_path/'full/artifacts.json').read_text())['files']
    assert files=={str(p.relative_to(tmp_path/'full')):q.o.g.ready.sha(p) for p in (tmp_path/'full').rglob('*') if p.is_file() and p.name!='artifacts.json'}
    assert 'NOT_REVIEWED' in (tmp_path/'full/human_review_template.csv').read_text(encoding='utf-8-sig')
    with pytest.raises(FileExistsError): q.run(q.parser().parse_args(['--out-dir',str(tmp_path/'full')]))


def test_failure_records_scope_and_preserves_already_rendered_evidence(monkeypatch,tmp_path):
    data=mock_input(tmp_path); data[-1]['slots']*=2
    monkeypatch.setattr(q,'checked_inputs',lambda *_:deepcopy(data)); monkeypatch.setattr(q.o.g.ready,'DATA',tmp_path)
    original=e.render_panel; calls=[]
    def fail_second(*args):
        calls.append(1)
        if len(calls)>1: raise RuntimeError('render failed')
        return original(*args)
    monkeypatch.setattr(e,'render_panel',fail_second)
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='render failed'): q.run(q.parser().parse_args(['--out-dir',str(out)]))
    completion=json.loads((out/'completion.json').read_text())
    assert completion['status']=='FAILED' and completion['rendered_images']==1
    assert completion['scope']['head_updates']==0 and len(list((out/'panels').glob('*.png')))==1
    assert json.loads((out/'artifacts.json').read_text())['status']=='FAILED'
