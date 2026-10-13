import hashlib
from pathlib import Path
import tempfile
import unittest
from crane_project.utils import port_closed_oof_cleanup_v1 as c


class CleanupTests(unittest.TestCase):
    def test_paths_and_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();outside=root/'outside';outside.mkdir()
            for name in ('../x','/tmp/x','work_dirs/crane_symeood_k1/epoch_24.pth'):
                with self.assertRaises(ValueError):c.allowed(root,name)
            base=root/'work_dirs/port_reliability_oof_v1'/c.RUNS[0];base.mkdir(parents=True)
            (base/'link').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):c.allowed(root,str((base/'link/a.pt').relative_to(root)))

    def test_all_validation_precedes_deletion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();base=root/'work_dirs/port_reliability_oof_v1'/c.RUNS[0];base.mkdir(parents=True)
            (base/'a.pt').write_bytes(b'a');(base/'b.pt').write_bytes(b'b')
            p=c.plan(root);(base/'b.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):c.execute(root,p,{})
            self.assertTrue((base/'a.pt').exists())

    def test_checkpoint_and_duplicate_only(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();base=root/'work_dirs/port_reliability_oof_v1'/c.RUNS[0];base.mkdir(parents=True)
            for name,blob in [('a.pt',b'weights'),('first.diff',b'diff'),('second.diff',b'diff'),('report.json',b'{}')]:
                (base/name).write_bytes(blob)
            formal=root/'work_dirs/formal';formal.mkdir();(formal/'epoch_24.pth').write_bytes(b'protected')
            p=c.plan(root);r=c.execute(root,p,{})
            self.assertEqual(len(r['removed']),2)
            self.assertTrue((base/'first.diff').exists());self.assertTrue((base/'report.json').exists())
            self.assertTrue((formal/'epoch_24.pth').exists())


if __name__=='__main__':unittest.main()
