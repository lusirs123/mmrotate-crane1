import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

P=Path(__file__).resolve().parents[1]/'crane_project/tools/cleanup_symeood_results_redundant_v1.py'
spec=importlib.util.spec_from_file_location('clean_results',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class DeletionGuards(unittest.TestCase):
    def test_active_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve()
            for name in [m.ACTIVE+'/queue.log','../outside.zip','/absolute.zip']:
                with self.assertRaises(AssertionError):m.allowed(root,name)

    def test_symlink_target_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();p=root/'work_dirs'/m.NAMES[0];p.mkdir(parents=True)
            f=p/'file.zip';f.symlink_to(root/'outside')
            with self.assertRaises(AssertionError):m.allowed(root,str(f.relative_to(root/'work_dirs')))

    def test_archive_member_hash_and_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'proof.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('result.json','evidence')
            self.assertEqual(list(m.archive_files(p)),[('result.json',8,hashlib.sha256(b'evidence').hexdigest())])
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','x')
            with self.assertRaises(AssertionError):list(m.archive_files(p))

    def test_changed_preserved_copy_prevents_all_deletion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();wd=root/'work_dirs';out=wd/m.OUT;out.mkdir(parents=True)
            arc=wd/m.NAMES[0]/'proof.zip';arc.write_bytes(b'archive')
            kept=wd/m.NAMES[0]/'result.json';kept.write_bytes(b'original')
            inventory=dict(config_guards={},protected_binary_files=[]);m.save(out/'inventory.json',inventory)
            plan=dict(guard_inventory_sha256=m.sha(out/'inventory.json'),delete_empty_dirs=[],preserve_archives=[],
                delete_files=[dict(path=str(arc.relative_to(wd)),bytes=arc.stat().st_size,sha256=m.sha(arc),
                    reason='archive_all_members_retained_by_sha256',retained=[dict(path=str(kept.relative_to(root)),sha256=m.sha(kept))])])
            m.save(out/'plan.json',plan);kept.write_bytes(b'changed')
            with self.assertRaises(AssertionError):m.execute(root)
            self.assertTrue(arc.exists())


if __name__=='__main__':unittest.main()
