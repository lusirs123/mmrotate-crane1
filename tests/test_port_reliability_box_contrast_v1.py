import unittest
from copy import deepcopy
import inspect
import numpy as np
from crane_project.utils import port_reliability_box_contrast_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple

try:
    import torch
except ImportError:
    torch = None


class NumericTests(unittest.TestCase):
    def setUp(self):
        self.box = [50.,40.,80.,20.,.37,.9]

    def test_generation_has_no_gt_argument(self):
        self.assertEqual(list(inspect.signature(core.training_boxes).parameters), ['pred'])

    def test_all_twelve_variants_and_identity(self):
        before=deepcopy(self.box); boxes,keys=core.training_boxes(self.box)
        self.assertEqual(self.box,before); self.assertEqual(boxes.shape,(13,6))
        self.assertEqual(keys[0],'M'); np.testing.assert_array_equal(boxes[0],self.box)
        np.testing.assert_array_equal(boxes[:,[0,1,4,5]],np.tile(np.array(self.box)[[0,1,4,5]],(13,1)))

    def test_raw_width_height_association(self):
        b=deepcopy(self.box);b[2:4]=[20.,80.]
        boxes,keys=core.training_boxes(b)
        self.assertEqual(boxes[keys.index('long_1.15'),3],92.)
        self.assertEqual(boxes[keys.index('long_1.15'),2],20.)
        self.assertTrue(np.all(boxes[:,3]>=boxes[:,2]))

    def test_reject_axis_inversion_instead_of_swap(self):
        b=deepcopy(self.box);b[2:4]=[21.,20.]
        boxes,keys=core.training_boxes(b)
        self.assertNotIn('long_0.85',keys); self.assertNotIn('short_1.15',keys)
        self.assertTrue(np.all(boxes[:,2]>=boxes[:,3]))

    def test_missing_has_no_candidates(self):
        boxes,keys=core.training_boxes(None)
        self.assertEqual(boxes.shape,(0,6)); self.assertEqual(keys,[])

    def test_invalid_box_rejected(self):
        b=deepcopy(self.box);b[2]=0.
        with self.assertRaises(ValueError):core.training_boxes(b)

    def test_labels_follow_gt_not_perturbation_sign(self):
        boxes,keys=core.training_boxes(self.box)
        small_gt=self.box[:5].copy();small_gt[2:4]=[68.,17.]
        large_gt=self.box[:5].copy();large_gt[2:4]=[92.,23.]
        a=core.offline_labels(boxes,small_gt);b=core.offline_labels(boxes,large_gt)
        self.assertEqual(a[keys.index('both_0.85')],0)
        self.assertEqual(b[keys.index('both_1.15')],0)
        self.assertEqual(a[keys.index('both_1.15')],1)
        self.assertEqual(b[keys.index('both_0.85')],1)

    def test_label_boundary_and_width_swap(self):
        b=np.array([[0,0,110,55,.1,.9],[0,0,110.01,55,.1,.9],[0,0,55,110,.1,.9]])
        np.testing.assert_array_equal(core.offline_labels(b,[0,0,100,50,.1]),[0,1,0])

    def test_pi_periodic_labels(self):
        b=np.array([self.box]);y=core.offline_labels(b,self.box[:5])
        b[0,4]+=np.pi
        np.testing.assert_array_equal(y,core.offline_labels(b,self.box[:5]))

    def test_spatial_pool_layout(self):
        a=np.arange(256*81,dtype=np.float32).reshape(1,256,9,9)
        value=core.pool_reference(a).reshape(256,3,3)
        self.assertEqual(value[7,2,1],a[0,7,6:9,3:6].mean())

    def test_bad_pool_shape_rejected(self):
        with self.assertRaises(ValueError):core.pool_reference(np.ones((1,32,9,9)))

    def test_original_train_class_weights_only(self):
        y=np.array([0.,0.,0.,1.]);w=core.class_weights(y)
        self.assertEqual(w[y==0].sum(),w[y==1].sum())

    def test_mechanism_pairs_ignore_padding(self):
        value=core.contrast_diagnostic([[1.,2.,-20.]],[[0,1,1]],[[True,True,False]])
        self.assertEqual(value['pairs'],1);self.assertEqual(value['pair_AUROC'],1.)

    def test_global_gate_checks_control(self):
        point=dict(actual=dict(states=dict(CR=95,FR=5,FA=3),runs=dict(correct_rejection=dict(longest=1))),
            control_tie_bounds={c:dict(bad_min_over_tie=4) for c in core.CONTROLS},
            matched_CR_controls={c:dict(states=dict(FA=4),exact_CR=True,runs=dict(correct_rejection=dict(longest=1))) for c in core.CONTROLS})
        self.assertEqual(core.gate(dict(all=point))['selected_arm'],'contrast')
        point['control_tie_bounds']['original']['bad_min_over_tie']=3
        result=core.gate(dict(all=point));self.assertFalse(result['passed'])
        self.assertIn('original',[v.get('control') for v in result['failures']])

    def test_online_decision_only_size_changes(self):
        d=dict(final_box_original=self.box,center_accepted=True,size_accepted=True,angle_accepted=False,
               risks=dict(size=.1,angle=.9))
        result=core.decide(d,.8,.5);self.assertFalse(result['size_accepted'])
        result['size_accepted']=True;result['risks']['size']=.1;self.assertEqual(result,d)

    def test_missing_not_rejected_or_classified(self):
        d=dict(final_box_original=None,center_accepted=False,size_accepted=False,angle_accepted=False)
        self.assertEqual(core.decide(d,None,.5),d)
        with self.assertRaises(ValueError):core.decide(d,.1,.5)

    def test_hidden_diagnostic_uses_259_dimension_contract(self):
        rows=[dict(image='a',pred=self.box,gt=self.box[:5],domain='real',sequence='seq01')]
        model={'network.0.weight':np.zeros((16,258)), 'network.0.bias':np.zeros(16),
               'network.2.weight':np.zeros((8,16)), 'network.2.bias':np.zeros(8),
               'network.4.weight':np.zeros((1,8)), 'network.4.bias':np.zeros(1)}
        value=core.hidden_diagnostic(rows,['a'],np.ones((1,259)),model,dict(mean=[0.]*259,scale=[1.]*259))
        self.assertEqual(value['all']['second_hidden_all_zero'],1)
        self.assertEqual(value['all']['zero_bad'],0)


@unittest.skipIf(torch is None,'Torch tests execute in server mmrotljj')
class TorchTests(unittest.TestCase):
    def setUp(self):
        from crane_project.utils import port_reliability_box_contrast_v1_torch as native
        self.native=native;self.models=native.make_models('cpu')
        torch.manual_seed(83)
        self.x=torch.randn(4,259);self.z=torch.tensor([-.5,.5,1.,2.])
        self.y=torch.tensor([0.,1.,0.,1.]);self.w=torch.ones(4)
        self.ax=torch.randn(4,12,259);self.az=self.z[:,None].expand(4,12).clone()
        self.ay=torch.tensor([[0.,1.]*6]*4);self.mask=torch.ones(4,12,dtype=torch.bool)

    def terms(self,arm):
        return self.native.losses(self.models[arm],self.x,self.z,self.y,self.w,self.ax,self.az,self.ay,self.mask,arm)

    def test_exact_neutral_same_capacity_and_start(self):
        self.assertEqual(self.native.exported(self.models['original']),self.native.exported(self.models['contrast']))
        for m in self.models.values():
            self.assertEqual(sum(p.numel() for p in m.parameters()),4290)
            self.assertTrue(torch.equal(m(self.x,self.z),-self.z))

    def test_control_ignores_auxiliary_labels(self):
        before=self.terms('original');self.ay[:]=1.
        after=self.terms('original')
        self.assertTrue(torch.equal(before[0],after[0]));self.assertEqual(float(after[1]),0.)

    def test_image_normalization_not_number_of_candidates(self):
        self.mask[:]=False;self.mask[0,:]=True;self.mask[1,0]=True
        self.ay[0,:]=0.;self.ay[1,0]=0.
        _,aux=self.terms('contrast')
        expected=.25*(torch.nn.functional.softplus(-self.z[0])+torch.nn.functional.softplus(-self.z[1]))/4
        torch.testing.assert_close(aux,expected)

    def test_padding_has_no_loss_or_gradient(self):
        self.mask[:,6:]=False
        _,first=self.terms('contrast')
        self.ax[:,6:]=float('nan');self.az[:,6:]=float('nan');self.ay[:,6:]=float('nan')
        _,second=self.terms('contrast');torch.testing.assert_close(first,second)

    def test_empty_auxiliary_has_zero_loss(self):
        self.mask[:]=False;_,aux=self.terms('contrast');self.assertEqual(float(aux),0.)

    def test_gradients_and_clip_and_numpy_replay(self):
        model=self.models['contrast'];opt=self.native.optimizer(model)
        records=[self.native.update(model,opt,self.x,self.z,self.y,self.w,self.ax,self.az,self.ay,self.mask,
                                    'contrast',component_gradients=True) for _ in range(2)]
        self.assertGreater(records[1]['components']['weighted_auxiliary_gradient_norm'],0.)
        self.assertGreater(records[1]['gradient_by_parameter']['network.0.weight'],0.)
        self.assertLessEqual(records[1]['gradient_norm_after'],5.0001)
        reference=core.numpy_logits(self.native.exported(model),self.x.numpy(),dict(mean=[0.]*259,scale=[1.]*259))
        # Risk z is supplied independently; put the same raw score into replay.
        reference+=self.x[:,0].numpy()-self.z.numpy()
        np.testing.assert_allclose(model(self.x,self.z).detach().numpy(),reference,atol=1e-6)

    def test_box_sampling_gt_free_and_coordinate_restore(self):
        from crane_project.utils import port_geometry_refine_g_v1 as geometry
        from crane_project.utils import port_reliability_box_contrast_v1_torch as native
        meta=dict(scale_factor=[.8,.8005,.8,.8005],img_shape=(640,800,3),ori_shape=(800,1000,3),
                  pad_shape=(1024,1024,3),flip=False)
        boxes=np.array([[300.,200.,80.,20.,.37,.9],[300.,200.,92.,23.,.37,.9]])
        p3=torch.arange(256*128*128,dtype=torch.float32).reshape(1,256,128,128)/10000
        sampled,support=native.box_features(p3,boxes,meta)
        self.assertEqual(sampled.shape,(2,2304));self.assertEqual(support.shape,(2,1,9,9))
        self.assertFalse(np.array_equal(sampled[0],sampled[1]))
        bt=torch.tensor(boxes[:,:5],dtype=torch.float32)
        restored=geometry.map_boxes(geometry.map_boxes(bt,meta),meta,inverse=True)
        torch.testing.assert_close(restored,bt)
        self.assertNotIn('gt',inspect.signature(native.box_features).parameters)

    def test_box_sampling_rejects_unfrozen_feature_graph(self):
        from crane_project.utils import port_reliability_box_contrast_v1_torch as native
        meta=dict(scale_factor=[1.]*4,img_shape=(64,64,3),ori_shape=(64,64,3),pad_shape=(64,64,3),flip=False)
        with self.assertRaises(ValueError):native.box_features(torch.ones(1,256,8,8,requires_grad=True),[[32,32,20,10,0,.9]],meta)


if __name__=='__main__':unittest.main()
