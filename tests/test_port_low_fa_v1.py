import unittest
from crane_project.tools.analyze_port_low_fa_v1 import curve,select,metrics

def row(i,wrong=False,risk=.1,gate=True,seq='real_x',missing=False):
    return dict(image=str(i),sequence=seq,frame_id=i,pred=None if missing else [0,0,15 if wrong else 10,5,0,.9],gt=[0,0,10,5,0],gate=gate,scores=dict(simple=risk,midpoint=risk,score=risk))
class TestLowFA(unittest.TestCase):
    def test_whole_tie(self):
        p=curve([row(0),row(1,True),row(2,risk=.2)],'simple')
        self.assertEqual([x['CR']+x['FA'] for x in p],[0,2,3])
        self.assertEqual(select(p,'CR',1,2,2,1)['FA'],1)
    def test_ed_no_split(self):
        p=curve([row(0),row(1,True)],'simple')
        self.assertIsNone(select(p,'ED',1,1,1,1)['cut'])
    def test_rational_target(self):
        p=[dict(ED=242,CR=5),dict(ED=243,CR=4)]
        self.assertEqual(select(p,'ED',243,342,5,342)['ED'],243)
    def test_fixed_gate_unreachable(self):
        p=curve([row(0,gate=False),row(1)],'temporal_fixed_gate')
        self.assertIsNone(select(p,'CR',95,100,2,0))
    def test_missing(self):
        r=[row(0),row(1,missing=True)]
        self.assertEqual(metrics(r,'simple',1)['states'],dict(FA=0,FR=0,ED=0,CR=1,MISSING=1))
    def test_runs(self):
        r=[row(0),row(2),row(3,missing=True),row(4),row(5,seq='real_y')]
        v=metrics(r,'simple',None)
        self.assertEqual(v['longest_FR_observations'],2)
        self.assertEqual(v['longest_correct_FR'],1)
        self.assertEqual(v['longest_without_CR_observations'],4)
    def test_exhaustive_curves(self):
        import itertools
        for labels in itertools.product((False,True),repeat=5):
            r=[row(i,wrong=b,risk=(i//2)/10,gate=i!=2) for i,b in enumerate(labels)]
            for method in ('simple','temporal_fixed_gate'):
                p=curve(r,method)
                for point in p:
                    v=metrics(r,method,point['cut'])['states']
                    for k in ('FA','FR','ED','CR'):self.assertEqual(v[k],point[k])
                g=labels.count(False);b=labels.count(True)
                for kind in ('CR','ED'):
                    got=select(p,kind,3,5,g,b)
                    valid=[x for x in p if x[kind]*5>=(g if kind=='CR' else b)*3]
                    self.assertEqual(got,(valid[0] if kind=='CR' else valid[-1]) if valid else None)
if __name__=='__main__':unittest.main()
