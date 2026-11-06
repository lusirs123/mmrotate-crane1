"""Efficiency must replay completed accuracy under the existing timing protocol."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crane_project.tools import benchmark_external_obb_v1 as e


class ExternalEfficiency(unittest.TestCase):
    def fixture(self, root):
        root=Path(root);test=root/'test/epoch_18';(test/'preds').mkdir(parents=True)
        (root/'selection').mkdir()
        (test/'preds/results.pkl').write_bytes(b'fixture; never unpickled')
        (test/'per_frame.jsonl').write_text('{}\n')
        config=e.config_identity('gaucho')
        lock=dict(model='gaucho',selected_completed_epoch=18,checkpoint='/weights/epoch_18.pth',
                  checkpoint_sha256='frozen-weight',configuration=config)
        def save(p,v):p.write_text(json.dumps(v))
        save(root/'selection/selected_identity.json',lock)
        save(root/'completion.json',dict(passed=True,TEST_runs=1,training_updates=0,
            TEST_reselection=False,selected_epoch=18,checkpoint_sha256='frozen-weight'))
        save(root/'protocol.json',dict(training_sources={}))
        save(test/'export_receipt.json',dict(passed=True,model='gaucho',split='test',frames=1440,
            frame_ids=list(range(1440)),checkpoint_sha256='frozen-weight',configuration=config,
            selection_lock_sha256=e.common.frozen.sha(root/'selection/selected_identity.json'),
            training_updates=0,initialized_state_sha256='state',final_state_sha256='state',
            results_sha256=e.common.frozen.sha(test/'preds/results.pkl')))
        save(test/'paper_metrics.json',dict(split='test',metrics=[dict(frames=1440)],
            native_checks=[dict(pass_check=True)]))
        return root,test

    def test_profile_pins_val_weight_and_common_protocol(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'validate_test_lock'):
            root,test=self.fixture(tmp);cfg=e.make_profile(root)
            self.assertEqual(cfg['warmup_frames'],50)
            self.assertEqual(cfg['repeats'],3)
            self.assertEqual(cfg['models']['gaucho']['weights'][0]['epoch'],18)
            for field,value in [('repeats',1),('warmup_frames',0),('physical_gpu_choices',[2,3])]:
                changed=deepcopy(cfg);changed[field]=value
                with self.assertRaises(ValueError):e.validate_profile(changed)
            changed=deepcopy(cfg);changed['models']['gaucho']['weights'][0]['epoch']=24
            with self.assertRaisesRegex(ValueError,'VAL-selected'):e.validate_profile(changed)

    def test_incomplete_accuracy_or_changed_predictions_cannot_start_timing(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'validate_test_lock'):
            root,test=self.fixture(tmp)
            (test/'preds/results.pkl').write_bytes(b'changed outputs')
            with self.assertRaisesRegex(ValueError,'outputs'):e.make_profile(root)
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'validate_test_lock'):
            root,test=self.fixture(tmp)
            done=e.common.read(root/'completion.json');done['TEST_runs']=2
            (root/'completion.json').write_text(json.dumps(done))
            with self.assertRaisesRegex(ValueError,'VAL-locked'):e.make_profile(root)

    def test_completed_accuracy_files_remain_unchanged_when_preparing_timing(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'validate_test_lock'):
            root,test=self.fixture(Path(tmp)/'accuracy')
            before={str(p):e.common.frozen.sha(p) for p in root.rglob('*') if p.is_file()}
            e.prepare(root,Path(tmp)/'efficiency')
            self.assertEqual(before,{str(p):e.common.frozen.sha(p) for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):e.prepare(root,Path(tmp)/'efficiency')


if __name__=='__main__':unittest.main()
