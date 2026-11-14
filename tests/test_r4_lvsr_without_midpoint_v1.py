"""B-only conditioning, no cache mutation, exact existing-head interfaces."""
from copy import deepcopy
import unittest
import torch
from crane_project.tools import train_r4_lvsr_without_midpoint_v1 as t
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils import port_size_qalign_v1 as vector
from tests.test_lvsr_conr_v1 import inputs


def record():
    x=inputs()
    x[5][:,2:4]*=1.04
    return dict(image='real_seq01_00001',sequence='real_seq01',domain='real',scale=1.,
        role='train',eligible=False,roi=x[0],support=x[1],boxes_original=x[2],
        boxes_model=x[3],scale_xy=x[4],midpoint_original=x[5],gt_original=x[6])


class WithoutMChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_fixed_existing_structure_and_recipe(self):
        cfg=t.checked_config()
        self.assertEqual(sum(p.numel() for p in PairSpecificSizeHead().parameters()),2480)
        self.assertEqual(cfg['updates'],24*639)
        self.assertFalse(cfg['TEST'])

    def test_condition_replaced_without_mutating_or_copying_features(self):
        source=record();old=source['midpoint_original'].clone();derived=t.condition_on_B([source])[0]
        self.assertIsNot(source,derived)
        self.assertTrue(torch.equal(source['midpoint_original'],old))
        self.assertTrue(torch.equal(derived['midpoint_original'],source['boxes_original']))
        self.assertNotEqual(derived['midpoint_original'].data_ptr(),source['boxes_original'].data_ptr())
        for key in ('roi','support','boxes_original','boxes_model','scale_xy','gt_original'):
            self.assertIs(derived[key],source[key],key)
        self.assertEqual((derived['image'],derived['scale'],derived['eligible']),('real_seq01_00001',1.,False))
        self.assertIsNone(derived['midpoint_original'].grad_fn)

    def test_shape_and_score_mismatch_rejected(self):
        r=record();r['midpoint_original']=r['midpoint_original'][:1]
        with self.assertRaises(ValueError):t.condition_on_B([r])
        r=record();r['midpoint_original'][:,5]-=.1
        with self.assertRaises(ValueError):t.condition_on_B([r])

    def test_neutral_and_empty_are_exact_B(self):
        r=t.condition_on_B([record()])[0];h=PairSpecificSizeHead().double()
        keys=('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')
        x=[r[k] for k in keys];out=h(*x)
        self.assertTrue(torch.equal(out['boxes_original'],r['boxes_original']))
        self.assertEqual(len(h(*[v[:0] for v in x])['boxes_original']),0)

    def test_loss_target_is_GT_over_B_not_GT_over_M20(self):
        r=record();b=t.condition_on_B([r])[0]
        before=vector.residual.size_targets(r['gt_original'],r['midpoint_original'],r['boxes_model'])['log_residual']
        after=vector.residual.size_targets(b['gt_original'],b['midpoint_original'],b['boxes_model'])['log_residual']
        expected=(b['gt_original'][:,2:4]/b['boxes_original'][:,2:4]).log()
        torch.testing.assert_close(after,expected)
        self.assertFalse(torch.equal(before,after))
        h=PairSpecificSizeHead().double();out=h(*[b[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')])
        torch.testing.assert_close(vector.size_loss(out,b['gt_original'],b['midpoint_original'],b['boxes_model']),
            torch.nn.functional.smooth_l1_loss(out['delta_roi'],expected,beta=.1))

    def test_nonzero_head_preserves_B_center_angle_score_and_learns(self):
        r=t.condition_on_B([record()])[0];h=PairSpecificSizeHead().double()
        with torch.no_grad():
            h.terminal.branches[0].bias.copy_(r['boxes_original'].new_tensor([.003,0.,-.003,0.]))
            h.terminal.branches[1].bias.copy_(r['boxes_original'].new_tensor([0.,.002,0.,-.002]))
        out=h(*[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')])
        self.assertTrue(torch.equal(out['boxes_original'][:,[0,1,4,5]],r['boxes_original'][:,[0,1,4,5]]))
        self.assertTrue(bool(out['accepted'].all()))
        self.assertFalse(torch.equal(out['boxes_original'][:,2:4],r['boxes_original'][:,2:4]))
        loss=vector.size_loss(out,r['gt_original'],r['midpoint_original'],r['boxes_model']);loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in h.parameters()))

    def test_no_stored_midpoint_geometry_leaks_into_forward(self):
        r=record();q=dict(r,midpoint_original=r['midpoint_original'].clone())
        q['midpoint_original'][:,:2]+=.3;q['midpoint_original'][:,2:4]*=1.02
        a=t.condition_on_B([r])[0];b=t.condition_on_B([q])[0];h=PairSpecificSizeHead().double()
        with torch.no_grad():
            h.terminal.branches[0].bias.fill_(.001)
        keys=('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')
        aa=h(*[a[k] for k in keys]);bb=h(*[b[k] for k in keys])
        for k in ('delta_roi','common_support','boxes_original'):self.assertTrue(torch.equal(aa[k],bb[k]),k)

    def test_anisotropic_model_axis_swap_neutral_B_and_restore(self):
        r=record();r['boxes_original'][:,2:4]=r['boxes_original'].new_tensor([61.,60.])
        r['scale_xy'][:]=r['scale_xy'].new_tensor([.49,.51])
        r['boxes_model']=r['boxes_original'][:,:5].clone()
        r['boxes_model'][:,:4]*=r['scale_xy'][:,[0,1,0,1]]
        r=t.condition_on_B([r])[0];h=PairSpecificSizeHead().double()
        out=h(*[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')])
        self.assertTrue(bool((r['boxes_model'][:,2]<r['boxes_model'][:,3]).all()))
        self.assertTrue(torch.equal(out['boxes_original'],r['boxes_original']))

    def test_four_paths_keep_B_and_M_center_metrics_separate(self):
        current=[];old=[]
        for i in range(887):
            b=[100.,100.,40.,20.,0.,.9];m=[101.,100.,40.,20.,0.,.9]
            row=dict(image=str(i),gt=b[:5],b=b[:],midpoint=b[:],edge_residual=b[:],
                domain='real' if i<375 else 'sim',sequence='real_seq07' if i<375 else 'sim_seq10')
            current.append(row);old.append(dict(row,midpoint=m,edge_residual=m))
        result=t.four_paths(current,old)
        self.assertEqual(result['overall']['R4_LB']['center_MAE_px'],0.)
        self.assertEqual(result['overall']['R4_M20_LM']['center_MAE_px'],1.)
        self.assertEqual(result['overall']['R4_LB']['joint10'],887)


if __name__=='__main__':unittest.main()
