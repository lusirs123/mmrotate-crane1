"""Resume validation and monitoring failure handling; no model or GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from crane_project.tools import resume_yolo26_native_v1 as resume


class ResumeSafety(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((resume.ROOT/resume.CONFIG).read_text())
        self.meta=dict(sha256=self.cfg['checkpoint_sha256'],epoch=3,completed_epochs=4,
            optimizer_present=True,optimizer_state_entries=444,ema_finite=True,
            head_class='OBB26',classes=1,version='8.4.202',scaler={},
            train_args=dict(task='obb',epochs=24,imgsz=1024,batch=2,nbs=4,amp=False,
                optimizer='MuSGD',seed=0,deterministic=True,nms=False,save_period=1))

    def test_complete_checkpoint_restarts_epoch_five_with_total_24(self):
        self.assertEqual(resume.validate_metadata(self.meta,self.cfg),dict(
            completed_epochs=4,next_epoch=5,total_epochs=24,remaining_epochs=20))

    def test_stripped_optimizer_wrong_checkpoint_or_recipe_rejected(self):
        changes=[dict(optimizer_present=False),dict(optimizer_state_entries=0),
            dict(sha256='different'),dict(epoch=4),dict(ema_finite=False)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                resume.validate_metadata(dict(self.meta,**change),self.cfg)
        for key,value in [('epochs',28),('imgsz',640),('batch',4),('amp',True)]:
            meta=deepcopy(self.meta);meta['train_args'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                resume.validate_metadata(meta,self.cfg)

    def test_transient_query_failure_retries_without_training_restart(self):
        query=Mock(side_effect=[subprocess.CalledProcessError(6,['nvidia-smi']),{'ok':True}])
        pause=Mock();errors=[]
        self.assertEqual(resume.retry_query(query,pause=pause,on_error=errors.append),{'ok':True})
        self.assertEqual(query.call_count,2);pause.assert_called_once_with(1)
        self.assertEqual(len(errors),1)

    def test_persistent_query_failure_saves_receipt_without_fourth_query(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            protocol=dict(resume_config=self.cfg,before={'last_good':True})
            (out/'protocol.json').write_text(json.dumps(protocol))
            child=Mock();child.pid=123;child.poll.return_value=None;child.wait.return_value=-15
            query=Mock(side_effect=subprocess.CalledProcessError(6,['nvidia-smi']))
            with patch.object(resume,'verify_protocol'),patch.object(resume.subprocess,'Popen',return_value=child),\
                    patch.object(resume,'snapshot',query),patch.object(resume.time,'sleep'),\
                    patch.object(resume.native,'stop_group') as stop:
                resume.controller(out)
            result=json.loads((out/'completion.json').read_text())
            self.assertFalse(result['passed']);self.assertEqual(result['state'],'STOPPED_OR_FAILED')
            self.assertEqual(result['gpu_query_errors'],3);self.assertEqual(query.call_count,3)
            self.assertEqual(result['last_successful_snapshot'],protocol['before'])
            self.assertFalse(result['automatic_training_retry']);stop.assert_called_once_with(child,out)

    def test_default_entry_only_audits_and_never_launches(self):
        with patch.object(resume.sys,'argv',['resume']),patch.object(resume,'audit',return_value={'passed':True}),\
                patch.object(resume,'launch') as launch,patch('builtins.print'),\
                patch.dict(resume.os.environ,{},clear=False):
            resume.main();launch.assert_not_called()
            self.assertEqual(resume.os.environ['CUDA_VISIBLE_DEVICES'],'')


if __name__=='__main__':unittest.main()
