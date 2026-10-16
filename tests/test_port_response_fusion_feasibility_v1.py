import unittest
from crane_project.tools import analyze_port_response_fusion_feasibility_v1 as a

def row(i,wrong,scores,present=True):
    return dict(image=str(i),sequence='real_seq01',domain='real',frame_id=i,split='train',image_size=[100,100],
        pred=[0,0,14 if wrong else 10,5,0,.9] if present else None,gt=[0,0,10,5,0],audit_risks=dict(zip(a.METHODS,scores)))

class FeasibilityTests(unittest.TestCase):
    def test_partition_missing_and_inclusive_tie(self):
        cuts={n:.5 for n in a.METHODS}
        self.assertEqual(a.cell(row(0,False,[.5]*5),cuts),'all_accept')
        self.assertEqual(a.cell(row(0,True,[.6]*5),cuts),'all_reject')
        self.assertEqual(a.cell(row(0,True,[.1,.9,.1,.9,.1]),cuts),'disagreement')
        self.assertEqual(a.cell(row(0,False,[None]*5,False),cuts),'missing')
    def test_known_pair_dominance_and_auc(self):
        data=[row(0,False,[.4]*5),row(1,True,[.6]*5),row(2,True,[.2]*5),row(3,True,[.4]*5),row(4,True,[.6,.2,.6,.2,.6])]
        d=a.evidence(data)
        self.assertEqual(d['pair_dominance'],dict(error_higher_all=1,error_lower_all=1,conflicting=1,identical=1))
        self.assertEqual(d['AUROC']['formal'],.625)
        self.assertEqual(d['AUROC']['bce_1701'],.375)
        self.assertIsNone(a.evidence(data[:1])['AUROC']['formal'])
    def test_accounting_intersection_oracle(self):
        cuts={n:.5 for n in a.METHODS}
        data=[row(0,False,[.2]*5),row(1,True,[.2]*5),row(2,True,[.2,.8,.8,.8,.8]),row(3,False,[.8,.2,.2,.2,.2]),row(4,True,[.8]*5),row(5,False,[None]*5,False)]
        v=a.analyze(data,cuts)['all']
        self.assertEqual(v['intersection']['states']['FA'],1);self.assertEqual(v['union']['states']['CR'],2)
        self.assertEqual(v['individual']['bce_1701']['changes_vs_simple'],dict(new_ED=1,new_FR=0,rescued_CR=1,released_FA=0))
        self.assertEqual(v['oracle_fixed_decision']['FA_lower_bound'],1)
        self.assertEqual(v['cells']['missing']['missing'],1)
