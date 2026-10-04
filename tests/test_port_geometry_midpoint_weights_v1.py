"""Fixed grid, validated text reuse, paired isolation and executed budgets.

Baseline fixtures are explicitly synthetic text/schema examples; they do not
claim to be real200-step training. Runtime tests use two real CPU updates/arm.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from crane_project.tools import preflight_port_geometry_midpoint_weights_v1 as t
from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
from crane_project.utils import port_geometry_midpoint_size_v1 as size
from crane_project.utils import port_geometry_midpoint_motion_v1 as motion

spec = importlib.util.spec_from_file_location('paired_fixtures',
    t.ROOT/'tests/test_port_geometry_midpoint_motion_v1.py')
helpers = importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def baseline_fixture(tmp_path):
    """Synthetic completed text evidence with the real schema/source chain."""
    torch.set_num_threads(1)
    train,cache,cache_doc = helpers.fake_contract(tmp_path)
    _,_,proof = t.paired.checked_contract(train,cache)
    records,blocks = t.base.select_blocks(helpers.synthetic_views())
    catalog = t.paired.pair_catalog(records);batches = t.paired.schedule(catalog)
    f.seed_all();head = f.m.SpatialMidpointHead()
    state = f.g.state_digest(head);initial = t.base.evaluate(head,records,torch.device('cpu'),f)
    results = {};events = [dict(stage='train_tensor_loaded',file=name) for name in ('train_s1.pt','train_s05.pt')]
    for arm in t.paired.ARMS:
        logs = [dict(stage='update',arm=arm,step=i+1,pair_indices=indices,point_loss=.01,
            motion_unit_loss=.02,motion_weighted_loss=.0005,motion_applied=arm=='point_motion',
            total_loss=.0105 if arm=='point_motion' else .01,grad_norm_before_clip=.1,
            grad_norm_after_clip=.1,stem_grad_norm=.001,
            component_gradients=dict(motion_unit_norm=.1) if i+1 in (1,2,25,100,200) else None)
            for i,indices in enumerate(batches)]
        results[arm] = dict(initial_state=state,final_state=state,initial=initial,final=initial,
            logs=logs,completed_updates=len(logs),checkpoint_exported=False,motion_gradient_signal_seen=True)
        for z in logs:
            events.extend([dict(stage='optimizer_step',arm=arm,step=z['step']),z])
    verdict = t.paired.review(initial,initial)
    verdict['checks']['measured_motion_gradient_signal_both_arms'] = True
    verdict['diagnostic_conditions_met'] &= True
    complete = dict(protocol=t.paired.VERSION,status='TRAIN_MOTION_SHORT_FIT_COMPLETE_REVIEW_REQUIRED',
        settings=deepcopy(t.paired.SETTINGS),proof=proof,runtime=cache_doc['runtime'],
        head_updates_total=2*t.SETTINGS['steps_per_arm'],cache_tensor_loads=2,
        detector_updates=0,detector_forward_calls=0,feature_extractions=0,
        val_access=False,test_access=False,weight_bytes_read=False,checkpoint_exported=False,
        formal_training_approved=False,automatic_promotion=False,pair_minibatches=batches,
        pair_support=catalog['support'],review=verdict,
        arm_states={a:dict(initial=r['initial_state'],final=r['final_state'],updates=r['completed_updates']) for a,r in results.items()})
    selection = dict(blocks=blocks,views=[{k:r[k] for k in ('image','sequence','domain','frame_id','role','scale','eligible')} for r in records])
    audit = t.paired.correspondence_audit(records,catalog,f,motion,torch)
    directory = tmp_path/'baseline';directory.mkdir()
    documents = {'completion.json':complete,'protocol.json':t.paired.protocol_document(),
        'sources.json':t.read_json(t.paired.SOURCES),'pair_catalog.json':catalog,
        'sample_selection.json':selection,'axis_correspondence.json':audit,'review.json':verdict}
    documents.update({a+'.json':dict(protocol=t.paired.VERSION,proof=proof,arm=a,result=r) for a,r in results.items()})
    for name,doc in documents.items():t.write_json(directory/name,doc)
    events.append(dict(stage='complete',status=complete['status']))
    (directory/'progress.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in events))
    t.paired.publish(directory)
    reused = dict(completion=complete,results=results,catalog=catalog,selection=selection,
        axis_audit=audit,files=dict(t.read_json(directory/'artifacts.json')['files'],
            **{'artifacts.json':t.sha(directory/'artifacts.json')}))
    return train,cache,cache_doc,proof,directory,reused


def rewrite(directory,name,value):
    (directory/name).write_text(json.dumps(value,allow_nan=False)+'\n')
    (directory/'artifacts.json').write_text(json.dumps(dict(protocol=t.paired.VERSION,
        files={n:t.sha(directory/n) for n in t.BASELINE_FILES})))


def test_fixed_seven_conditions_no_weight_combinations_or_automatic_selection():
    protocol = t.protocol_document()
    assert protocol['weights'] == [.0125,.025,.05]
    assert len(t.CONDITIONS) == 7 and sum(c['reused'] is None for c in t.CONDITIONS) == 5
    assert protocol['scope']['new_head_updates_total'] == 1000
    assert protocol['scope']['reused_historical_updates'] == 400
    assert protocol['scope']['represented_updates'] == 1400
    assert not protocol['scope']['formal_training_approved'] and not protocol['scope']['test_access']


@pytest.mark.parametrize('relative',[True,False])
def test_python38_real_static_cli_no_torch_tensor_or_gpu_and_no_overwrite(tmp_path,relative):
    train,cache,_,_,baseline,_ = baseline_fixture(tmp_path)
    script = Path(t.__file__).resolve();target = str(script.relative_to(t.ROOT)) if relative else str(script)
    out = tmp_path/'static'
    cmd = [sys.executable,'-B',target,'--check-only','--training-dir',str(train),
        '--cache-dir',str(cache),'--baseline-dir',str(baseline),'--out-dir',str(out)]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.run(cmd,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert proc.returncode == 0,proc.stderr
    result = t.read_json(out/'completion.json')
    assert result['status'] == 'STATIC_WEIGHTS_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
    assert result['head_updates_total'] == result['cache_tensor_loads'] == 0 and result['gpu_devices_used'] == []
    content = {p.name:p.read_bytes() for p in out.iterdir()}
    again = subprocess.run(cmd,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert again.returncode != 0 and 'existing results preserved' in again.stderr
    assert content == {p.name:p.read_bytes() for p in out.iterdir()}


def test_static_reuse_contract_never_imports_runtime_and_source_tampering_aborts(tmp_path,monkeypatch):
    train,cache,_,_,baseline,_ = baseline_fixture(tmp_path)
    def forbidden(*a,**k):raise AssertionError('Static runtime import')
    monkeypatch.setattr(t,'runtime_modules',forbidden)
    args = t.parser().parse_args(['--check-only','--training-dir',str(train),'--cache-dir',str(cache),
        '--baseline-dir',str(baseline),'--out-dir',str(tmp_path/'static')])
    assert t.run(args)['head_updates_total'] == 0
    bad = deepcopy(t.read_json(t.SOURCES));bad['sources'][str(Path(t.__file__).resolve().relative_to(t.ROOT))] = 'bad'
    path = tmp_path/'bad_sources.json';t.write_json(path,bad);monkeypatch.setattr(t,'SOURCES',path)
    with pytest.raises(ValueError,match='Source SHA'):t.checked_contract(train,cache,baseline)


@pytest.mark.parametrize('bad',['sha','formula','batch','identity','runtime','old_size_protocol','review','extra_update','gradient_flag'])
def test_incompatible_reuse_is_rejected_without_silent_retraining(tmp_path,bad):
    _,_,cache,proof,baseline,_ = baseline_fixture(tmp_path)
    if bad == 'sha':
        (baseline/'point_only.json').write_text('{}')
    elif bad in ('formula','batch','identity','old_size_protocol','gradient_flag'):
        doc = t.read_json(baseline/'point_only.json')
        if bad == 'formula':doc['result']['logs'][0]['total_loss'] += .1
        elif bad == 'batch':doc['result']['logs'][0]['pair_indices'] = [999]*4
        elif bad == 'identity':doc['proof']['cache_manifest_sha256'] = 'wrong'
        elif bad == 'old_size_protocol':doc['protocol'] = t.base.VERSION
        else:doc['result']['motion_gradient_signal_seen'] = False
        rewrite(baseline,'point_only.json',doc)
    elif bad == 'runtime':
        doc = t.read_json(baseline/'completion.json');doc['runtime'] = {'wrong':True};rewrite(baseline,'completion.json',doc)
    elif bad == 'review':
        doc = t.read_json(baseline/'review.json');doc['diagnostic_conditions_met'] = True;rewrite(baseline,'review.json',doc)
    else:
        path = baseline/'progress.jsonl';path.write_text(path.read_text()+json.dumps(dict(stage='optimizer_step',arm='extra',step=1))+'\n')
        rewrite(baseline,'completion.json',t.read_json(baseline/'completion.json'))
    with pytest.raises(ValueError):t.audit_baseline(baseline,proof,cache)


def synthetic_run_setup(tmp_path,monkeypatch):
    train,cache,document,_,baseline,reused = baseline_fixture(tmp_path)
    # A two-step CPU execution fixture, not a changed production protocol.
    monkeypatch.setitem(t.SETTINGS,'steps_per_arm',2);monkeypatch.setitem(t.paired.SETTINGS,'steps_per_arm',2)
    reused['completion']['pair_minibatches'] = reused['completion']['pair_minibatches'][:2]
    monkeypatch.setattr(t,'checked_contract',lambda *a:(t.protocol_document(),document,{'synthetic':True},reused))
    monkeypatch.setattr(t.base,'execution_device',lambda *a:(torch.device('cpu'),document['runtime']))
    monkeypatch.setattr(t.base,'load_train_cache',lambda *a:helpers.synthetic_views())
    args = t.parser().parse_args(['--training-dir',str(train),'--cache-dir',str(cache),
        '--baseline-dir',str(baseline),'--out-dir',str(tmp_path/'run')])
    return args,reused


def test_five_real_two_step_arms_share_batches_formulas_support_and_export_no_weights(tmp_path,monkeypatch):
    args,reused = synthetic_run_setup(tmp_path,monkeypatch)
    report = t.run(args)
    assert report['head_updates_total'] == 10 and report['reused_historical_updates'] == 4 and report['represented_updates'] == 14
    assert report['status'] == 'TRAIN_WEIGHTS_GRID_COMPLETE_REVIEW_REQUIRED'
    assert report['selected_condition'] is None and not report['formal_training_approved'] and not report['test_access']
    out = Path(args.out_dir);assert not list(out.glob('*.pth'))
    assert not (out/'point_only.json').exists() and not (out/'motion_w0p025.json').exists()
    hashes = []
    for c in t.CONDITIONS:
        if c['reused']:continue
        result = t.read_json(out/(c['name']+'.json'))['result'];hashes.append(result['final_state']['parameters'])
        assert result['initial'] == reused['results']['point_only']['initial']
        for z in result['logs']:
            unit = z['size_unit_loss'] if c['family'] == 'size' else z['motion_unit_loss']
            assert z['total_loss'] == pytest.approx(z['point_loss']+c['weight']*unit,abs=1e-8)
            assert z['pair_indices'] == reused['completion']['pair_minibatches'][z['step']-1]
            assert z['component_gradients'][c['family']]['unit_norm'] > 0
        assert result['final']['groups']['probe/real/1.0']['temporal_pairs'] == 14
    assert len(set(hashes)) == 5
    rows = t.read_json(out/'summary.json')['rows'];assert len(rows) == 56
    assert {r['condition'] for r in rows} == {c['name'] for c in t.CONDITIONS}
    assert all(r['frames'] == 16 and r['output_frames'] == 16 and r['conditional_center_correct_count'] == r['all_frame_center_correct_count'] == 16 for r in rows if r['role'] == 'probe')
    assert (out/'summary.csv').exists() and (out/'reviews.json').exists()


def test_mismatch_aborts_before_updates_and_partial_failure_preserves_completed_arm(tmp_path,monkeypatch):
    args,reused = synthetic_run_setup(tmp_path,monkeypatch)
    reused['results']['point_only']['initial']['rows'][0]['gt'][0] += 1.
    with pytest.raises(ValueError,match='initial outputs'):t.run(args)
    assert t.read_json(Path(args.out_dir)/'completion.json')['head_updates_total'] == 0
    reused['results']['point_only']['initial']['rows'][0]['gt'][0] -= 1.
    original = t.fit_condition
    def fail_second(*a,**k):
        if a[4]['name'] == 'size_w0p025':raise ValueError('synthetic second arm failure')
        return original(*a,**k)
    monkeypatch.setattr(t,'fit_condition',fail_second);args.out_dir = str(tmp_path/'partial')
    with pytest.raises(ValueError,match='second arm'):t.run(args)
    report = t.read_json(Path(args.out_dir)/'completion.json')
    assert report['head_updates_total'] == 2 and report['represented_updates'] == 6
    assert (Path(args.out_dir)/'size_w0p0125.json').exists() and (Path(args.out_dir)/'artifacts.json').exists()


def test_executed_nonfinite_update_has_accurate_failure_budget(tmp_path,monkeypatch):
    args,_ = synthetic_run_setup(tmp_path,monkeypatch)
    original = torch.optim.Adam.step
    def broken(optimizer,*a,**k):
        original(optimizer,*a,**k)
        with torch.no_grad():optimizer.param_groups[0]['params'][0].fill_(float('nan'))
    monkeypatch.setattr(torch.optim.Adam,'step',broken)
    with pytest.raises(ValueError,match='Nonfinite updated'):t.run(args)
    report = t.read_json(Path(args.out_dir)/'completion.json')
    assert report['head_updates_total'] == 1 and report['represented_updates'] == 5


def test_valid_zero_auxiliary_gradient_completes_without_promoting(tmp_path,monkeypatch):
    args,_ = synthetic_run_setup(tmp_path,monkeypatch)
    original = motion.motion_loss
    def zero(*a):
        value,detail = original(*a)
        return value*0.,dict(detail,parts=detail['parts']*0.)
    monkeypatch.setattr(motion,'motion_loss',zero)
    report = t.run(args);assert report['head_updates_total'] == 10
    reviews = t.read_json(Path(args.out_dir)/'reviews.json')
    assert all(not reviews[c['name']]['checks']['active_auxiliary_gradient_signal_seen'] for c in t.CONDITIONS if c['family'] == 'motion' and not c['reused'])


def test_immutability_no_parameter_loss_changes_and_low_dfr_alone_not_a_pass(tmp_path):
    _,_,_,_,_,reused = baseline_fixture(tmp_path)
    control = reused['results']['point_only'];changed = deepcopy(control)
    changed['active_auxiliary_gradient_signal_seen'] = True
    changed['final']['groups']['probe/sim/0.5']['midpoint']['dfr_pct_per_frame'] = 0.
    assert not t.condition_review(control,changed,t.CONDITIONS[1])['diagnostic_conditions_met']
    assert size.SIZE_WEIGHT == motion.MOTION_WEIGHT == .025
    assert f.m.SETTINGS['prior_sigma_cells'] == 1. and f.m.SETTINGS['roi_size'] == 9
