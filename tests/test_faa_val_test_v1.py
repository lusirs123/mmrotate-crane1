"""Identity gates for the frozen FAA VAL -> TEST transition; no GPU/data use."""
from copy import deepcopy
import unittest

from crane_project.tools import eval_faa_val_test_v1 as e


def fixture():
    rule = dict(r_center_delta=.005, mcml_limit=5)
    protocol = dict(sources={e.CONFIG: 'config-sha'}, selection_config=rule)
    entries = {'epoch_%d'%i: dict(checkpoint=str(e.checkpoint(i)), checkpoint_sha256=e.WEIGHTS[i])
               for i in e.EPOCHS}
    record = dict(candidate_epochs=list(e.EPOCHS), all_checkpoints=entries,
                  config_sha256='config-sha', selection_config=deepcopy(rule),
                  center_thresh_px=15., evidence_role='source_val_checkpoint_selection',
                  selected_checkpoint='epoch_20', selected_path=str(e.checkpoint(20)))
    return record, protocol


class FAASelectionGate(unittest.TestCase):
    def test_accepts_complete_frozen_selection(self):
        self.assertEqual(e.validate_selection(*fixture()), 20)

    def test_partial_val_cannot_unlock_test(self):
        record, protocol = fixture()
        del record['all_checkpoints']['epoch_18']
        with self.assertRaisesRegex(ValueError, 'candidate set'):
            e.validate_selection(record, protocol)

    def test_rule_config_weight_or_selected_path_substitution_rejected(self):
        changes = (
            lambda r: r['selection_config'].update(mcml_limit=6),
            lambda r: r.update(config_sha256='another-config'),
            lambda r: r['all_checkpoints']['epoch_22'].update(checkpoint_sha256='0'*64),
            lambda r: r.update(selected_path=str(e.checkpoint(24))),
        )
        for change in changes:
            record, protocol = fixture(); change(record)
            with self.assertRaises(ValueError):
                e.validate_selection(record, protocol)


if __name__ == '__main__':
    unittest.main()
