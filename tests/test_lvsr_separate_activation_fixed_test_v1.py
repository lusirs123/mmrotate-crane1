"""Fixed representative identity and shared TEST metric semantics."""
from copy import deepcopy
import unittest
from crane_project.tools import eval_lvsr_separate_activation_ep16_test_v1 as e
from crane_project.tools import train_lvsr_separate_activation_v1 as t


def fixture():
    cp=dict(path='head_epoch_16.pth',sha256=e.SHA,epoch=16)
    events=[dict(epoch=i,eligibility_vs_L5=dict(pass_all=False),selection_key=[0 if i!=16 else -1,i],
        checkpoint=cp if i==16 else dict(path='other',sha256='other',epoch=i)) for i in range(1,25)]
    summary=dict(selection=t.choose(events),epochs=events,TEST=False)
    complete=dict(passed=True,epochs_completed=24,formal_optimizer_updates=15336,checkpoint=cp,TEST=False)
    return summary,complete


class FixedTestChecks(unittest.TestCase):
    def test_accepts_predeclared_representative_without_rewriting_null(self):
        s,c=fixture();e.validate_training(s,c)
        self.assertIsNone(s['selection']['selected_epoch'])
        self.assertFalse(s['selection']['selection_on_test'])

    def test_rejects_epoch_sha_or_ranking_reselection(self):
        for field,value in (('epoch',17),('sha256','wrong'),('path','head_epoch_17.pth')):
            s,c=fixture();s=deepcopy(s);s['selection']['representative_checkpoint'][field]=value
            with self.assertRaises(ValueError):e.validate_training(s,c)
        s,c=fixture();s['epochs'][-1]['selection_key']=[-999,24]
        with self.assertRaises(ValueError):e.validate_training(s,c)

    def test_metric_keeps_missing_output_in_full_frame_denominator(self):
        rr=[dict(image='sim_a_%05d'%i,sequence='sim_a',frame_id=i,domain='sim',gt=[0,0,100,50,0],
            prediction=[0,0,100,50,0,.9] if i==0 else None) for i in range(2)]
        m,_,_,_,_=e.metric(rr,'prediction')
        self.assertEqual(m['frames'],2);self.assertEqual(m['outputs'],1)
        self.assertEqual(m['joint10'],1);self.assertEqual(m['RIoU'],.5)
        self.assertEqual(m['long_MAE_pct'],0.);self.assertEqual(m['short_MAE_pct'],0.)
        self.assertEqual(m['MCML_max'],1)


if __name__=='__main__':unittest.main()
