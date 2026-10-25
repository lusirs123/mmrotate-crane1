"""CPU-only safety gates; no torch/MMCV or GPU needed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
CHECK=ROOT/'crane_project/tools/native_symeood_b111_entry_check_v1.py'
VERIFY=ROOT/'crane_project/tools/verify_symeood_canonical_cpu_v1.py'
SPEC=importlib.util.spec_from_file_location('canonical_verify',VERIFY)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class CpuPreparationGates(unittest.TestCase):
    def test_native_entry_default_does_not_import_training(self):
        result=subprocess.run([sys.executable,str(CHECK)],text=True,capture_output=True)
        self.assertEqual(result.returncode,0)
        self.assertIn('NOT_EXECUTED',result.stdout)
        self.assertNotIn('Traceback',result.stderr)

    def test_incomplete_execute_request_fails_before_importing_training(self):
        result=subprocess.run([sys.executable,str(CHECK),'--execute-native-check'],text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('AssertionError',result.stderr)
        self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_corrupt_cpu_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);r=base/'receipts';r.mkdir()
            p=r/'CPU_complete.json';p.write_text('{"passed":true}')
            (r/'receipt_manifest.json').write_text(json.dumps({p.name:'incorrect-hash'}))
            with self.assertRaises(AssertionError):MODULE.verify(base)


if __name__=='__main__':unittest.main()
