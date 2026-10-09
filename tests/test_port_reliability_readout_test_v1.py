import inspect
import json
from copy import deepcopy
from pathlib import Path
import gzip
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np

from crane_project.utils import port_reliability_readout_test_v1 as core
from crane_project.utils import port_reliability_state_continuity_v1 as old_states
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.tools import eval_port_reliability_readout_test_v1 as runner
from crane_project.tools.review_port_reliability_readout_compare_v1 import independent_logits
from crane_project.tools import review_port_reliability_readout_test_v1 as review

try:
    import torch
    from crane_project.utils import port_reliability_readout_test_v1_torch as native
except ImportError:
    torch=None


def row(i,bad=False,missing=False):
    pred=None if missing else [0.,0.,12. if bad else 10.,4.,0.,.8]
    d=dict(final_box_original=pred,center_accepted=not missing,size_accepted=not missing,
           angle_accepted=not missing,risks=dict(size=None if missing else .2,angle=None if missing else .3))
    return dict(image='real_seq03_%05d'%i,sequence='real_seq03',frame_id=i,domain='real',split='test',
        pred=pred,gt=[0.,0.,10.,4.,0.],image_size=[100,80],angle_axis_well_defined=True,
        image_sha256='img',annotation_sha256='ann',original_simple_decision=d)


def zero_model():
    return {'network.0.weight':np.zeros((16,290)).tolist(),'network.0.bias':np.zeros(16).tolist(),
        'network.2.weight':np.zeros((8,16)).tolist(),'network.2.bias':np.zeros(8).tolist(),
        'network.4.weight':np.zeros((1,8)).tolist(),'network.4.bias':[0.],'beta':[0.]}


def bundle():
    model=dict(protocol=core.scoring.VERSION,epoch=100,update_counts=dict(temperature=1000,residual=1000),
        normalizer=dict(mean=np.zeros(291).tolist(),scale=np.ones(291).tolist()),
        models={m:zero_model() for m in core.scoring.ARMS})
    cutoffs={m:dict(single_global_cutoff=True,risk_le=.5) for m in ('temperature','residual','full_simple','score_only')}
    report=dict(protocol=core.scoring.VERSION,fixed_final_epoch=100,update_counts=deepcopy(model['update_counts']),
        TEST_read=False,original_policy_changed=False,boxes_scores_output_center_angle_unchanged=True,
        status='VAL_FAILED_STOP',gate=dict(passed=False),VAL_cutoffs=deepcopy(cutoffs))
    return model,cutoffs,report


class NumericContract(unittest.TestCase):
    def test_failed_VAL_frozen_bundle_not_promoted(self):
        a,b,c=bundle();before=deepcopy((a,b,c));x,y=core.frozen_bundle(a,b,c)
        x['epoch']=1;y['temperature']['risk_le']=0.
        self.assertEqual((a,b,c),before)

    def test_reject_epoch_updates_and_cutoff_changes(self):
        for key,value in [('epoch',99),('update_counts',dict(temperature=999,residual=1000))]:
            a,b,c=bundle();a[key]=value
            with self.assertRaises(ValueError):core.frozen_bundle(a,b,c)
        a,b,c=bundle();b['residual']['risk_le']=.9
        with self.assertRaises(ValueError):core.frozen_bundle(a,b,c)

    def test_reject_rewritten_VAL_history(self):
        for key,value in [('TEST_read',True),('status','VAL_PASSED'),('original_policy_changed',True)]:
            a,b,c=bundle();c[key]=value
            with self.assertRaises(ValueError):core.frozen_bundle(a,b,c)

    def test_scores_no_fit_or_gate_and_only_size_changes(self):
        a,b,_=bundle();rows=[row(1),row(2,bad=True),row(3,missing=True)]
        x=np.zeros((2,291));x[:,0]=[2.,-2.];before=deepcopy((a,b,rows))
        with patch.object(core.scoring,'gate',side_effect=AssertionError('No TEST gate')):
            risks,decisions=core.score_rows(rows,x,[r['image'] for r in rows[:2]],a,b)
        self.assertEqual((a,b,rows),before)
        self.assertLess(risks['residual'][rows[0]['image']],.5)
        self.assertGreater(risks['residual'][rows[1]['image']],.5)
        for r,d in zip(rows,decisions):
            for m,v in d['methods'].items():
                self.assertEqual(v['center_accepted'],r['original_simple_decision']['center_accepted'])
                self.assertEqual(v['angle_accepted'],r['original_simple_decision']['angle_accepted'])
                self.assertEqual(v['final_box_original'],r['pred'])
        self.assertEqual(decisions[-1]['methods']['residual'],rows[-1]['original_simple_decision'])

    def test_scoring_independent_of_GT_labels(self):
        a,b,_=bundle();r=[row(1)];x=np.ones((1,291));ids=[r[0]['image']]
        first=core.score_rows(r,x,ids,a,b);r[0]['gt']=[999.,999.,1.,1.,2.]
        self.assertEqual(first,core.score_rows(r,x,ids,a,b))
        self.assertNotIn('gt',inspect.signature(core.scoring.decide).parameters)

    def test_feature_membership_and_finite_guard(self):
        a,b,_=bundle();r=[row(1)];x=np.zeros((1,291))
        with self.assertRaises(ValueError):core.score_rows(r,x,['other'],a,b)
        x[0,1]=np.nan
        with self.assertRaises(ValueError):core.score_rows(r,x,[r[0]['image']],a,b)

    def test_numpy_readout_independent_replay(self):
        a,_,_=bundle();x=np.random.RandomState(8).normal(size=(3,291))
        for arm in core.scoring.ARMS:
            np.testing.assert_allclose(core.scoring.numpy_logits(a['models'][arm],x,a['normalizer'],arm),
                independent_logits(a['models'][arm],x,a['normalizer'],arm)[0],atol=1e-12)

    def test_TEST_states_FR_ED_missing_and_original_guard(self):
        rows=[row(1,bad=True),row(2),row(4),row(5,missing=True)]
        v=core.test_summary(rows,set())
        self.assertEqual(v['states'],dict(FA=0,FR=2,ED=1,CR=0,MISSING=1))
        self.assertEqual(v['runs']['correct_rejection']['longest'],1)
        with self.assertRaises(ValueError):old_states.summarize(rows,set())
        with self.assertRaises(ValueError):core.test_summary(rows,{rows[-1]['image']})

    def test_fixed_points_centers_and_full_frame_denominators(self):
        a,b,_=bundle();rows=[row(1),row(2,bad=True),row(3,missing=True)]
        risk,_=core.score_rows(rows,np.zeros((2,291)),[r['image'] for r in rows[:2]],a,b)
        d=core.summarize(rows,risk,b)
        self.assertEqual(d['center']['all'],dict(frames=3,outputs=2,hits=2,output_coverage=2/3,
            hit_rate_on_outputs=1.,correct_coverage_all_frames=2/3))
        self.assertEqual(set(d['fixed_VAL_cutoff_summary']['all']),set(b))
        self.assertFalse(d['automatic_promotion']);self.assertFalse(d['thresholds_from_TEST'])
        self.assertEqual(d['original_policy_summary']['all']['states']['FA'],1)

    def test_same_count_tie_bounds_and_whole_tie_CR(self):
        rows=[row(1),row(2),row(3,bad=True)]
        scores={m:{r['image']:.2 for r in rows} for m in ('residual','temperature','full_simple','score_only')}
        scores['residual']={rows[0]['image']:.1,rows[1]['image']:.8,rows[2]['image']:.9}
        d=core.test_describe(rows,scores,'residual',.5)['all']
        v=d['same_count_controls']['score_only'];self.assertEqual(v['accepted_outputs'],1)
        self.assertEqual(v['partial_boundary_tie']['selection'],'image_order_only_not_GT')
        m=d['matched_CR_controls']['score_only'];self.assertEqual(m['states']['CR'],2)
        self.assertFalse(m['exact_CR']);self.assertTrue(m['whole_boundary_tie'])

    def test_pi_periodic_replay_and_exact_score(self):
        p=row(1)['pred'];q=p.copy();q[4]+=np.pi
        core.replay_box(p,q);q[5]+=.00001
        with self.assertRaises(ValueError):core.replay_box(p,q)
        with self.assertRaises(ValueError):core.replay_box(None,p)
        q=p.copy();q[2]+=.1
        with self.assertRaises(ValueError):core.replay_box(p,q)

    def test_full_TEST_identity_pairing_and_tamper(self):
        records=[];predictions=[]
        for seq,n in sorted(core.COUNTS.items()):
            for i in range(1,n+1):
                r=row(i);r.update(sequence=seq,image='%s_%05d'%(seq,i),domain=seq.split('_')[0])
                records.append(dict(r,methods=dict(simple=r['original_simple_decision'])))
                predictions.append(dict(image=r['image'],frame_id=i,image_sha256='img',original_size_hw=[80,100],
                    midpoint=r['pred'],b=r['pred'],original_b_raw_exact_before_after=True))
        self.assertEqual(len(core.bind_rows(records,predictions)),1440)
        predictions[0]['midpoint']=deepcopy(predictions[0]['midpoint']);predictions[0]['midpoint'][0]+=1
        with self.assertRaises(ValueError):core.bind_rows(records,predictions)
        with self.assertRaises(ValueError):core.bind_rows(records[:-1],predictions[:-1])

    def test_order_duplicates_and_untouched_roles(self):
        with self.assertRaises(ValueError):core.test_order([row(1),row(1)])
        r=row(1);r['split']='val'
        with self.assertRaises(ValueError):core.test_order([r])

    def test_TEST_statistics_match_original_contract_on_synthetic_rows(self):
        rows=[row(1),row(2,bad=True),row(4),row(5,missing=True)]
        accepted={rows[0]['image'],rows[1]['image']}
        expected=deepcopy(rows)
        for r in expected:r['split']='val'
        self.assertEqual(core.test_summary(rows,accepted),old_states.summarize(expected,accepted))

    def test_independent_audit_synthetic_end_to_end(self):
        a,b,_=bundle();rows=[row(1),row(2,bad=True),row(3,missing=True)]
        for r in rows:r['b_original']=r['pred']
        x=np.zeros((2,291))
        for i,r in enumerate(rows[:2]):x[i,:3]=simple.descriptor(r['pred'],r['image_size'])
        ids=[r['image'] for r in rows[:2]];risks,decisions=core.score_rows(rows,x,ids,a,b)
        report=dict(core.summarize(rows,risks,b),frozen_cutoffs=b,sources={},parent_VAL_status='VAL_FAILED_STOP',
            VAL_failure_retracted=False,TEST_used_for_selection=False,input_sha256=runner.PINS,
            GT_online=False,boxes_scores_center_angle_output_unchanged=True,
            detector_updates=0,head_updates=0,quality_updates=0,threshold_updates=0)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'report.json').write_text(json.dumps(report))
            np.savez_compressed(out/'test_features.npz',features=x,images=np.asarray(ids))
            with gzip.open(out/'scored_TEST.jsonl.gz','wt') as f:
                for r,v in zip(rows,decisions):f.write(json.dumps(dict(r,
                    experiment_risks={m:s[r['image']] for m,s in risks.items()},candidate_decisions=v['methods']))+'\n')
            receipt=dict(TEST_evaluated=True,automatic_promotion=False,protocol=core.VERSION,
                status=report['status'],artifacts={p.name:runner.sha(p) for p in out.iterdir()})
            (out/'completion.json').write_text(json.dumps(receipt))
            with patch.object(runner,'checked_inputs',return_value=(a,b,None,{})), \
                 patch.object(runner,'checked_sources',return_value=({},{})), \
                 patch.object(runner,'read_rows',return_value=[]), \
                 patch.object(core,'bind_rows',return_value=rows):
                result=review.audit(out)
            self.assertTrue(result['passed']);self.assertEqual(result['outputs'],2)

    def test_source_and_frozen_artifact_checks_without_TEST_evaluation(self):
        protocol,source=runner.checked_sources();a,b,_,proof=runner.checked_inputs(False)
        self.assertEqual(protocol['input_pins'],runner.PINS)
        self.assertEqual(source['sources'][str(runner.PROTOCOL.relative_to(runner.ROOT))],runner.sha(runner.PROTOCOL))
        self.assertEqual(a['epoch'],100);self.assertEqual(proof['parent_VAL_status'],'VAL_FAILED_STOP')
        self.assertFalse(proof['TEST_evaluated']);self.assertEqual(set(b),{'temperature','residual','full_simple','score_only'})

    def test_server_entry_has_no_training_and_one_archive(self):
        text=(runner.ROOT/'tools/run_port_reliability_readout_test_v1.sh').read_text()
        self.assertIn('for device in 2 3',text);self.assertNotIn('--mode train',text)
        self.assertIn('tar -czf "$archive"',text)
        self.assertIn('work_dirs/port_reliability_readout_test_v1_',text)

    def test_refused_existing_output_does_not_write_failure_into_history(self):
        with tempfile.TemporaryDirectory() as d:
            fake_root=Path(d);out=fake_root/'work_dirs'/core.VERSION/'old_run'/'result'
            out.mkdir(parents=True);(out/'sentinel').write_text('old result')
            with patch.object(runner,'ROOT',fake_root),patch.object(runner,'checked_sources',return_value=({},{})), \
                 patch.object(sys,'argv',['eval','--mode','check','--out',str(out)]):
                with self.assertRaises(FileExistsError):runner.main()
            self.assertEqual([p.name for p in out.iterdir()],['sentinel'])


@unittest.skipIf(torch is None,'Torch unavailable locally; native replay pending on server')
class TorchContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.b=row(1)['pred'];self.scale=[.5,.5001]
        self.record=dict(roi=torch.ones(1,256,9,9),support=torch.ones(1,1,9,9),
            boxes_original=torch.tensor([self.b],dtype=torch.float64),boxes_model=torch.tensor([self.b[:5]]),
            scale_xy=torch.tensor([self.scale],dtype=torch.float64),midpoint_original=torch.tensor([self.b],dtype=torch.float64))

    def test_GT_and_attachment_or_shape_rejected(self):
        native.validate_record(self.record,self.b,self.b,self.scale)
        bad=dict(self.record,gt=torch.zeros(1,5))
        with self.assertRaises(ValueError):native.validate_record(bad,self.b,self.b,self.scale)
        bad=dict(self.record,roi=self.record['roi'].clone().requires_grad_())
        with self.assertRaises(ValueError):native.validate_record(bad,self.b,self.b,self.scale)

    def test_actual_sx_sy_and_native_identity(self):
        with self.assertRaises(ValueError):native.validate_record(self.record,self.b,self.b,[.5,.5])
        bad=dict(self.record,boxes_original=self.record['boxes_original']+1)
        with self.assertRaises(ValueError):native.validate_record(bad,self.b,self.b,self.scale)

    def test_frozen_stem_features_missing_and_no_GT(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__();self.stem=torch.nn.Conv2d(256,32,1)
            def forward(self,roi,support,b,model,scale):
                if len(b):self.stem(roi)
                return dict(boxes_original=b)
        head=Head().eval().requires_grad_(False)
        empty={k:v[:0] for k,v in self.record.items()}
        inventory=[dict(image='a',b=self.b,midpoint=self.b,image_size=[100,80],scale_xy=self.scale),
                   dict(image='b',b=None,midpoint=None,image_size=[100,80],scale_xy=self.scale)]
        x,ids,proof=native.extract(head,[self.record,empty],inventory,'cpu')
        self.assertEqual(x.shape,(1,291));self.assertEqual(ids,['a']);self.assertEqual(proof['head_updates'],0)
        inventory[0]['gt']=[0,0,1,1,0]
        with self.assertRaises(ValueError):native.extract(head,[self.record,empty],inventory,'cpu')


if __name__=='__main__':unittest.main()
