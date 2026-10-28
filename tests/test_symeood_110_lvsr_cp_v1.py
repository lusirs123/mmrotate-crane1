"""CP-specific CPU autograd, recovery, selection and launch guards."""
from copy import deepcopy
import io
import subprocess
import sys
import unittest
from pathlib import Path
import torch
from crane_project.utils import symeood_110_lvsr_cp_v1 as cp
from crane_project.utils import port_size_a20_protection_train_v1 as original
from crane_project.tools import run_symeood_110_lvsr_cp_v1 as tool

torch.set_num_threads(1)


def output_fixture(active=True):
    z=torch.tensor([[.05,-.04],[.02,.03]],dtype=torch.float64,requires_grad=True)
    m=torch.tensor([[0.,0.,100.,50.,0.,.9]]*2,dtype=torch.float64)
    bm=m[:,:5].clone();gt=bm.clone().requires_grad_(True)
    candidate=torch.cat((m[:,:2],m[:,2:4]*z.exp(),m[:,4:]),-1)
    out=dict(delta_roi=z,candidate_original=candidate,vector_valid=torch.ones(2,dtype=torch.bool),candidate_valid=torch.ones(2,dtype=torch.bool))
    refs=dict(active=torch.full((2,2),active,dtype=torch.bool),reference=torch.full((2,2),.01,dtype=torch.float64))
    return z,m,bm,gt,out,refs


def head_fixture():
    b=torch.tensor([[100.,100.,100.,50.,.2,.9],[100.,100.,100.,50.,.2,.8]])
    xy=torch.tensor([[1.,1.],[.5,.5]]);bm=b[:,:5].clone();bm[:,:4]*=xy[:,[0,1,0,1]]
    m=b.clone();gt=b[:,:5].clone();gt[:,2]*=.97;gt[:,3]*=1.02
    x=[torch.randn(2,256,9,9),torch.ones(2,1,9,9),b,bm,xy,m,gt]
    refs=dict(active=torch.ones(2,2,dtype=torch.bool),reference=torch.zeros(2,2,dtype=torch.float64))
    return x,refs


def eligible_fixture():
    summary={};control={};parent={};near={};damages={};cd={}
    def cell(edges,joint,a=.01,r=.01,riou=.92):
        return dict(frames=10,failure_intervals=dict(riou=[]),sizes=dict(joint10=joint,edges={e:dict(mae=x) for e,x in zip(('long','short'),edges)},a=dict(abs_mae=a),r=dict(abs_mae=r)),mean_riou_full_frame=riou)
    for d in ('overall','real','sim'):
        m=cell([.08,.08] if d=='real' else [.05,.05],5,.03,.03,.90)
        c=cell([.065,.065] if d=='real' else [.035,.035],8)
        p=cell([.07,.07] if d=='real' else [.04,.04],7,.02,.02,.91)
        ctl=cell([.066,.066] if d=='real' else [.039,.039],7,.015,.015)
        b=cell([.06,.06],6,.02,.02,.91)
        temp=dict(dfr_percent_per_frame=1.,mcml_max=2,mcml_mean=1.,tdr_w10=dict(numerator=8,denominator=8),aci=.95,aci_adjacent_pairs=9)
        m['temporal']={d:deepcopy(temp)};c['temporal']={d:deepcopy(temp)}
        summary[d]=dict(midpoint=m,edge_residual=c,b=b,paired=dict(frozen_identity=True,new_output=0,lost_output=0));parent[d]=dict(edge_residual=p);control[d]=dict(edge_residual=ctl)
        near[d]=dict(near={e:dict(before_mae=.01,after_mae=.01) for e in ('long','short')})
        damages[d]=dict(long_near_positive_harm_mean_pct=.1,short_near_positive_harm_mean_pct=.1,long_near_material_harmed_0p1pp=1 if d!='overall' else 2)
        cd[d]=dict(long_near_positive_harm_mean_pct=.2,short_near_positive_harm_mean_pct=.2,long_near_material_harmed_0p1pp=2 if d!='overall' else 4)
    return summary,near,damages,control,cd,parent


class CPChecks(unittest.TestCase):
    def test_total_gradient_and_no_ratio_supervision(self):
        z,m,bm,gt,out,refs=output_fixture();v=cp.losses(out,gt,m,bm,refs,torch)
        total=torch.autograd.grad(v['total'],z,retain_graph=True)[0]
        size=torch.autograd.grad(v['size'],z,retain_graph=True)[0]
        protect=torch.autograd.grad(v['protect'],z,retain_graph=True)[0]
        self.assertTrue(torch.allclose(total,size+.1*protect,atol=1e-12,rtol=1e-12))
        self.assertGreater(float(v['ratio']),0.)
        self.assertGreater(float(protect[0,0]),0.);self.assertLess(float(protect[0,1]),0.)
        v['total'].backward();self.assertIsNone(gt.grad)

    def test_inactive_zero_hinge_equals_C_and_uses_all_edges(self):
        z,m,bm,gt,out,refs=output_fixture(active=False);v=cp.losses(out,gt,m,bm,refs,torch)
        a=torch.autograd.grad(v['total'],z,retain_graph=True)[0];b=torch.autograd.grad(v['size'],z,retain_graph=True)[0]
        self.assertTrue(torch.equal(a,b));self.assertEqual(float(v['protect']),0.)
        refs['active'][0,0]=True;v=cp.losses(out,gt,m,bm,refs,torch)
        self.assertAlmostEqual(float(v['protect']),max(float(z[0,0].exp()-1)-.01,0.)/4,places=12)

    def test_reference_gradients_rejected_not_reclassified(self):
        z,m,bm,gt,out,refs=output_fixture();refs['reference'].requires_grad_(True)
        with self.assertRaises(ValueError):cp.losses(out,gt,m,bm,refs,torch)

    def test_real_head_cpu_update_and_optimizer_recovery(self):
        torch.manual_seed(1703);x,refs=head_fixture()
        from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
        a=AlignedSizeHead('fixed_b');opt=torch.optim.Adam(a.parameters(),lr=.0001,weight_decay=0.)
        cp.update(a,opt,x,refs,torch)
        stream=io.BytesIO();torch.save(dict(head=a.state_dict(),opt=opt.state_dict()),stream);stream.seek(0);saved=torch.load(stream,map_location='cpu')
        b=AlignedSizeHead('fixed_b');b.load_state_dict(saved['head']);other=torch.optim.Adam(b.parameters(),lr=.0001,weight_decay=0.);other.load_state_dict(saved['opt'])
        self.assertEqual(cp.update(a,opt,x,refs,torch),cp.update(b,other,x,refs,torch))
        self.assertTrue(tool.equal(a.state_dict(),b.state_dict(),torch));self.assertTrue(tool.equal(opt.state_dict(),other.state_dict(),torch))
        self.assertTrue(torch.equal(a(*x[:-1])['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]]))

    def test_frozen_gates_reject_harm_transfer_and_equal_sim_long(self):
        s,n,d,c,cd,p=eligible_fixture();self.assertTrue(cp.eligibility(s,n,d,c,cd,p)['pass_all'])
        d['sim']['short_near_positive_harm_mean_pct']=.3
        self.assertIn('sim/short_positive_harm_vs_C',cp.eligibility(s,n,d,c,cd,p)['failed'])
        d['sim']['short_near_positive_harm_mean_pct']=.1
        s['sim']['edge_residual']['sizes']['edges']['long']['mae']=c['sim']['edge_residual']['sizes']['edges']['long']['mae']
        self.assertIn('sim/long_strict_C',cp.eligibility(s,n,d,c,cd,p)['failed'])

    def test_joint_first_selection_and_no_eligible_is_null(self):
        e=[dict(epoch=1,selection_key=[-8,.001,.01,.01,1],eligibility=dict(pass_all=True)),dict(epoch=2,selection_key=[-9,.002,.02,.02,2],eligibility=dict(pass_all=True))]
        self.assertEqual(cp.select(e)['selected'],2)
        for v in e:v['eligibility']['pass_all']=False
        self.assertIsNone(cp.select(e)['selected']);self.assertEqual(cp.select(e)['representative'],2)

    def test_source_control_and_default_no_train(self):
        proof=tool.checked_sources();self.assertEqual(proof['settings']['epochs'],6)
        present=[dict(image='frame%05d'%i,scale=1.,midpoint_original=[1]) for i in range(5112)]
        self.assertEqual(len(tool.check_schedule(present,proof)[1]),6)
        changed=deepcopy(proof);changed['control']['epoch_schedule_sha256']['1']='bad'
        with self.assertRaises(ValueError):tool.check_schedule(present,changed)
        result=subprocess.run([sys.executable,str(Path(tool.__file__))],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('NO_GPU',result.stdout)
        result=subprocess.run([sys.executable,str(Path(tool.__file__)),'--stage','train'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('Explicit --run required',result.stderr)


if __name__=='__main__':unittest.main()
