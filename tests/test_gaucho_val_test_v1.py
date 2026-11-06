"""CPU-only checks for GPU routing and the actual VAL -> lock -> TEST controller."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from crane_project.tools import eval_gaucho_val_test_v1 as workflow
from crane_project.tools.export_external_obb_v1 import gpu_binding


class GauChoEvaluation(unittest.TestCase):
    def test_authorized_gpu_pair_and_logical_mapping(self):
        devices=[dict(index=i,pci='%02d'%i) for i in (2,0,3,1)]
        self.assertEqual(gpu_binding(devices,'1',(0,1),'cuda:0')['index'],1)
        self.assertEqual(gpu_binding(devices,'0,1',(0,1),'cuda:1')['index'],1)
        self.assertEqual(gpu_binding(devices,'3',(2,3),'cuda:0')['index'],3)
        for visible,pair,logical in (('2',(0,1),'cuda:0'),('0,0',(0,1),'cuda:0'),
            ('0,00',(0,1),'cuda:0'),('GPU-uuid',(0,1),'cuda:0'),
            ('0',(0,1),'cuda:1'),('0,2',(0,2),'cuda:0')):
            with self.subTest(binding=visible),self.assertRaises(ValueError):
                gpu_binding(devices,visible,pair,logical)

    def test_exclusive_test_marker_preserves_first_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'TEST_execution.json'
            workflow.reserve_test(path,dict(checkpoint_sha256='original'))
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                workflow.reserve_test(path,dict(checkpoint_sha256='replacement'))
            self.assertEqual(path.read_bytes(),before)

    def run_controller(self, folder, fail_val=False, fail_lock=False):
        events=[]
        out=Path(folder)/'evaluation';out.mkdir();(out/'val').mkdir()
        weight=Path(folder)/'epoch_20.pth';weight.write_bytes(b'fixture only')
        digest=hashlib.sha256(weight.read_bytes()).hexdigest()
        protocol=dict(weights={str(e):digest for e in workflow.EPOCHS},
                      configuration={},training_protocol_sha256='training-sha')
        (out/'protocol.json').write_text(json.dumps(protocol))
        marker=Path(folder)/'TEST_execution.json'
        def infer(directory,jobs,spec):
            if jobs[0][0]=='test':
                self.assertTrue(marker.exists())
                self.assertIn('select',events)
                events.append(('test',jobs[0][1]))
            else:
                events.extend(('val',j[1]) for j in jobs)
                if fail_val:raise RuntimeError('VAL export incomplete')
        def evaluate(directory):
            events.append(('metrics',directory.parent.name,int(directory.name.split('_')[-1])))
        def select(model,val_dir,selection_dir):
            self.assertEqual([e[1] for e in events if e[0]=='val'],list(workflow.EPOCHS))
            self.assertEqual(len([e for e in events if e[0]=='metrics']),5)
            events.append('select');selection_dir.mkdir()
            (selection_dir/'selected_identity.json').write_text(json.dumps(dict(
                selected_completed_epoch=20,checkpoint=str(weight),locked_Taipei='fixture')))
        metrics_module=types.ModuleType('crane_project.tools.evaluate_external_obb_saved_v1')
        metrics_module.evaluate=evaluate
        selection_module=types.ModuleType('crane_project.tools.select_external_obb_val_v1')
        selection_module.select=select
        efficiency_module=types.ModuleType('crane_project.tools.benchmark_external_obb_v1')
        def efficiency(evaluation,efficiency_out,gpu):
            self.assertTrue((evaluation/'completion.json').exists())
            self.assertEqual(events[-1],('metrics','test',20))
            self.assertEqual(gpu,0)
            events.append('efficiency')
        efficiency_module.run=efficiency
        def validate_lock(*args):
            events.append('lock_checked')
            if fail_lock:raise ValueError('Invalid VAL selection lock')
        with ExitStack() as stack:
            stack.enter_context(patch.dict('os.environ'))
            stack.enter_context(patch.dict('sys.modules',{
                metrics_module.__name__:metrics_module,selection_module.__name__:selection_module,
                efficiency_module.__name__:efficiency_module}))
            for name,value in (('verify',lambda p:None),('infer',infer),
                               ('validate_test_lock',validate_lock),
                               ('gpu_rows',lambda:[]),('TEST_MARKER',marker)):
                stack.enter_context(patch.object(workflow,name,value))
            if fail_val:
                with self.assertRaisesRegex(RuntimeError,'incomplete'):workflow.controller(out)
            elif fail_lock:
                with self.assertRaisesRegex(ValueError,'Invalid VAL'):workflow.controller(out)
            else:workflow.controller(out)
        return out,events,marker

    def test_all_val_metrics_and_lock_precede_one_selected_test(self):
        with tempfile.TemporaryDirectory() as folder:
            out,events,marker=self.run_controller(folder)
            self.assertEqual([e for e in events if isinstance(e,tuple) and e[0]=='test'],[('test',20)])
            self.assertLess(events.index('lock_checked'),events.index(('test',20)))
            self.assertEqual(json.loads((out/'completion.json').read_text())['TEST_runs'],1)
            self.assertEqual(events[-1],'efficiency')
            self.assertTrue(json.loads((out/'pipeline_completion.json').read_text())['efficiency_complete'])

    def test_failed_val_never_selects_or_starts_test(self):
        with tempfile.TemporaryDirectory() as folder:
            out,events,marker=self.run_controller(folder,True)
            self.assertNotIn('select',events)
            self.assertFalse(marker.exists())
            self.assertFalse((out/'test').exists())
            self.assertEqual(json.loads((out/'status.json').read_text())['state'],'FAILED')

    def test_invalid_val_lock_never_reserves_or_starts_test(self):
        with tempfile.TemporaryDirectory() as folder:
            out,events,marker=self.run_controller(folder,fail_lock=True)
            self.assertIn('select',events)
            self.assertFalse(marker.exists())
            self.assertFalse((out/'test').exists())
            self.assertFalse((out/'completion.json').exists())


if __name__=='__main__':unittest.main()
