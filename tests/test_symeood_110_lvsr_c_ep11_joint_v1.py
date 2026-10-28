"""Selection revision and scoped reuse of the original C training loop."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from crane_project.tools import run_symeood_110_lvsr_c_ep11_joint_v1 as run


def event(epoch, count, long=.05, short=.05):
    summary = dict(overall=dict(edge_residual=dict(frames=887,
        joint_size10_full_frame=dict(numerator=count, denominator=887),
        output_coverage=dict(numerator=885),
        edges=dict(long=dict(mae=long), short=dict(mae=short))),
        paired=dict(frozen_identity=True, new_output=0, lost_output=0)))
    return dict(epoch=epoch, metrics=summary, selection_key=run.selection_key(summary, epoch),
                eligibility=dict(pass_all=False))


class RevisedC(unittest.TestCase):
    def test_joint_priority_not_original_eligibility_or_single_edge(self):
        events = [event(1, 650, .02, .02), event(2, 680, .06, .06)]
        self.assertEqual(run.choose(events)['selected'], 2)

    def test_tie_break_and_parent_retention(self):
        parent = event(0, 677)['metrics']
        self.assertFalse(run.select_with_parent([event(1, 676, .01, .01)], parent)['replace_parent'])
        self.assertFalse(run.select_with_parent([event(1, 677)], parent)['replace_parent'])
        self.assertTrue(run.select_with_parent([event(1, 677, .04, .05)], parent)['replace_parent'])
        self.assertEqual(run.choose([event(2, 680), event(1, 680)])['selected'], 1)

    def test_output_loss_cannot_be_hidden_by_joint_count(self):
        summary = deepcopy(event(1, 680)['metrics'])
        summary['overall']['paired']['lost_output'] = 1
        with self.assertRaises(ValueError):
            run.selection_key(summary, 1)

    def test_bindings_restore_after_failure_and_math_unchanged(self):
        names = ('ranking', 'choose', 'backup', 'update_status')
        original = {name: getattr(run.base, name) for name in names}
        update, evaluate, fit = run.old.update, run.old.evaluate, run.base.fit_stage
        with self.assertRaises(RuntimeError):
            with run.revised_reporting():
                self.assertIs(run.base.ranking, run.selection_key)
                self.assertIs(run.old.update, update)
                self.assertIs(run.old.evaluate, evaluate)
                self.assertIs(run.base.fit_stage, fit)
                raise RuntimeError('simulate loop failure')
        for name, value in original.items():
            self.assertIs(getattr(run.base, name), value)

    def test_original_two_argument_status_callback(self):
        with patch.object(run, 'status') as status:
            with run.revised_reporting():
                run.base.update_status(run.OUT, 'training', stage_name='C', epoch=0)
            status.assert_called_once_with('training', stage_name='C', epoch=0)


if __name__ == '__main__':
    unittest.main()
