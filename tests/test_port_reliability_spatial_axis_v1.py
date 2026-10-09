import unittest,inspect
import numpy as np
from crane_project.utils import port_reliability_spatial_axis_v1 as c

class SpatialTests(unittest.TestCase):
    def test_targets_swap_and_sign(self):
        a=dict(pred=[0,0,11,18,0,.9],gt=[0,0,10,20,0])
        np.testing.assert_allclose(c.targets([a]),[[-1,1]])
        a['pred'][2:4]=[18,11];np.testing.assert_allclose(c.targets([a]),[[-1,1]])
    def test_targets_missing(self):
        with self.assertRaises(ValueError):c.targets([dict(pred=None)])
    def test_coarse_block_layout(self):
        x=np.arange(81,dtype=float).reshape(1,1,9,9).repeat(256,1);z=c.coarse(x)
        for i in range(3):
            for j in range(3):np.testing.assert_array_equal(z[0,0,i*3:i*3+3,j*3:j*3+3],x[0,0,i*3:i*3+3,j*3:j*3+3].mean())
    def test_coarse_removes_detail(self):
        x=np.zeros((1,256,9,9));x[:,:,0,0]=1;x[:,:,0,1]=-1
        np.testing.assert_array_equal(c.coarse(x),0)
    def test_normalizer_train_only(self):
        with self.assertRaises(ValueError):c.fit_normalizer(np.zeros((1,256,9,9)),np.ones((1,1,9,9)),np.zeros((1,2)),'VAL')
    def test_unsupported_zero(self):
        x=np.ones((2,256,9,9));s=np.ones((2,1,9,9));s[:,:,0]=0;d=np.zeros((2,2));n=c.fit_normalizer(x,s,d,'TRAIN')
        x[:,:,0]=999;z,_=c.normalize(x,s,d,n);np.testing.assert_array_equal(z[:,:,0],0)
    def test_finite_and_shape(self):
        with self.assertRaises(ValueError):c.validate_inputs(np.zeros((1,256,8,8)),np.ones((1,1,9,9)),np.zeros((1,2)))
    def test_neutral_exact(self):
        a=np.linspace(.01,.99,200);np.testing.assert_array_equal(c.readout(a,np.zeros(200)),a)
    def test_decision_protection(self):
        a=dict(center_accepted=True,size_accepted=True,angle_accepted=False,risks=dict(size=.2,angle=.4),final_box_original=[1,2,3,4,0,.7])
        d=c.decide(a,.9,.5);self.assertFalse(d['size_accepted'])
        for k in ['center_accepted','angle_accepted','final_box_original']:self.assertEqual(a[k],d[k])
    def test_missing_not_judged(self):
        a=dict(center_accepted=False,size_accepted=False,angle_accepted=False,risks=dict(size=None,angle=None),final_box_original=None)
        self.assertEqual(c.decide(a,None,.5),a)
    def test_spatial_online_signature(self):
        self.assertNotIn('gt',inspect.signature(c.numpy_forward).parameters)
    def test_conv_reference(self):
        rng=np.random.RandomState(1);x=rng.randn(2,3,5,5);w=rng.randn(4,3,3,3);b=rng.randn(4);z=c.conv(x,w,b,1);xp=np.pad(x,((0,0),(0,0),(1,1),(1,1)))
        for n in range(2):
            for o in range(4):
                for i in range(5):
                    for j in range(5):self.assertAlmostEqual(z[n,o,i,j],np.sum(xp[n,:,i:i+3,j:j+3]*w[o])+b[o],places=11)
    def test_class_weights(self):
        y=np.r_[np.zeros(99),1.];w=c.class_weights(y);self.assertAlmostEqual(w[y==1].sum(),w[y==0].sum(),places=12)
    def test_real_labels(self):
        from crane_project.tools.run_port_reliability_spatial_axis_v1 import parts
        rows,_=parts();t=c.targets(rows['TRAIN']);self.assertEqual(int((np.abs(t).max(1)>1).sum()),71)
    def test_collector_global_names_resolve(self):
        import builtins,symtable
        from pathlib import Path
        from crane_project.tools import run_port_reliability_spatial_axis_v1 as run
        code=Path(run.__file__).read_text();table=symtable.symtable(code,run.__file__,'exec')
        collect=next(t for t in table.get_children() if t.get_name()=='collect')
        unresolved=[s.get_name() for s in collect.get_symbols() if s.is_global() and s.is_referenced() and s.get_name() not in vars(run) and not hasattr(builtins,s.get_name())]
        self.assertEqual(unresolved,[])
    def test_frozen_B_source_reference(self):
        from crane_project.tools.run_port_reliability_spatial_axis_v1 import historical
        self.assertEqual(historical.PINS[historical.historical.B_PATH],'8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23')
    def test_aux_threshold_contract(self):
        self.assertEqual(c.SETTINGS['auxiliary_weight'],.25);self.assertEqual(c.SETTINGS['parameter_count'],13115)

if __name__=='__main__':unittest.main()
