import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from crane_project.utils import external_training_monitor_io_v1 as monitor


class MonitorIOTests(unittest.TestCase):
    def test_briefly_missing_heartbeat_recovers_and_records_path(self):
        errors = io.StringIO()
        with patch.object(Path, 'read_text', side_effect=[FileNotFoundError(2, 'missing', 'rank_0_heartbeat.json'), '{"loss":0.7}']), \
                patch.object(monitor.time, 'sleep'), contextlib.redirect_stderr(errors):
            result = monitor.read_json('rank_0_heartbeat.json')
        self.assertEqual(result, {'loss': 0.7})
        event = json.loads(errors.getvalue())
        self.assertEqual(event['target'], 'rank_0_heartbeat.json')
        self.assertIn('FileNotFoundError', event['traceback'])

    def test_persistent_failure_is_bounded_and_fatal(self):
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError(2, 'missing')) as read, \
                patch.object(monitor.time, 'sleep') as sleep, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                monitor.read_json('missing.json')
        self.assertEqual(read.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    def test_partial_json_recovers(self):
        with patch.object(Path, 'read_text', side_effect=['{', '{"pid":42}']), \
                patch.object(monitor.time, 'sleep'), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(monitor.read_json('identity.json'), {'pid': 42})

    def test_replace_failure_retries_without_leaving_temporary_files(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'status.json'
            replace = monitor.os.replace
            calls = []

            def fail_once(source, dest):
                calls.append(source)
                if len(calls) == 1:
                    raise FileNotFoundError(2, 'temporary failure', source)
                return replace(source, dest)

            with patch.object(monitor.os, 'replace', side_effect=fail_once), \
                    patch.object(monitor.time, 'sleep'), contextlib.redirect_stderr(io.StringIO()):
                monitor.atomic_json(target, {'state': 'RUNNING'})
            self.assertEqual(json.loads(target.read_text()), {'state': 'RUNNING'})
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_gpu_timeout_recovers_with_bounded_subprocess_timeout(self):
        with patch.object(monitor.subprocess, 'check_output', side_effect=[subprocess.TimeoutExpired('nvidia-smi', 5), '2, id, pci, 8192, 7000, 1192, 100\n']) as query, \
                patch.object(monitor.time, 'sleep'), contextlib.redirect_stderr(io.StringIO()):
            rows = monitor.gpu_rows()
        # High memory is passed through unchanged for the existing guard to stop.
        self.assertEqual(rows[0]['used_mib'], 7000)
        self.assertEqual(rows[0]['free_mib'], 1192)
        self.assertEqual(query.call_args.kwargs['timeout'], 5)

    def test_programming_errors_are_not_retried(self):
        with patch.object(monitor.time, 'sleep') as sleep:
            with self.assertRaises(ValueError):
                monitor.retry_io('parse', 'gpu', lambda: int('invalid'))
        sleep.assert_not_called()

    def test_existing_memory_guard_still_stops_training(self):
        from crane_project.tools import run_gaucho_no_port_shrink_v1 as control
        from unittest.mock import Mock
        rows = [dict(index=i, uuid='gpu%d'%i, used_mib=7010, free_mib=1182) for i in (2, 3)]
        child = Mock(pid=99)
        child.poll.return_value = None
        child.wait.return_value = -15
        protocol = dict(ordinals=[2, 3], driver_mib=6656, minimum_free_mib=1536)
        original_protocol = deepcopy(protocol)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(control, 'assert_sources'), patch.object(control, 'idle_gpus'), \
                patch.object(control.subprocess, 'Popen', return_value=child), \
                patch.object(control, 'gpu_rows', return_value=rows), \
                patch.object(control, 'worker_pids', return_value={42}), \
                patch.object(control, 'processes', return_value=[dict(uuid='gpu2', pid=42, used_mib=7000)]), \
                patch.object(control, 'stop_group') as stop, patch.object(control, 'atomic', monitor.atomic_json), \
                patch.object(control, 'read', monitor.read_json):
            self.assertFalse(control.run_one('gaucho', Path(folder), protocol))
            stop.assert_called_once()
            result = json.loads((Path(folder)/'gaucho/completion.json').read_text())
            self.assertIn('MEMORY_GUARD', result['reason'])
        self.assertEqual(protocol, original_protocol)


if __name__ == '__main__':
    unittest.main()
