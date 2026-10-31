"""Native CPU-only tests and model-config audit; never launches GPU/training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only check requires hidden GPUs'
    import torch
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    protocol = json.loads((args.base / 'protocol.json').read_text())
    for relative, digest in protocol['sources'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    config = ROOT / protocol['config']
    assert hashlib.sha256(config.read_bytes()).hexdigest() == protocol['config_sha256']
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from crane_project.utils.eood_B_regression_protocol_v1 import check_model_delta
    cfg = Config.fromfile(str(config))
    import_modules_from_strings(**cfg.custom_imports)
    old_path = ROOT / protocol['baseline']['reference'] / 'crane_eood_k1_port_day2night_aug_b_v1.py'
    old = Config.fromfile(str(old_path))
    check_model_delta(old.model.to_dict(), cfg.model.to_dict())
    for key in old:
        if key not in ('model','custom_imports','work_dir'):
            assert old[key] == cfg[key], key
    assert cfg.custom_hooks == old.custom_hooks
    assert cfg.runner.max_epochs == 24 and cfg.data.samples_per_gpu == 2
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_eood_B_regression_v1.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun == 6 and not result.skipped
    assert not torch.cuda.is_initialized()
    receipt = dict(passed=True, native_tests=6, skipped=0, CPU_only=True,
        GPU_initialized=False, full_model_forward=False,
        note='Native CPU assigner, head forward and target-path tests; no real-image model inference, optimizer or training.',
        formal_training_started=False, optimizer_updates=0, TEST=False,
        control_retrain=False, resource_hold=True, GPU_gate_pending=True,
        config_sha256=protocol['config_sha256'],
        protocol_sha256=hashlib.sha256((args.base / 'protocol.json').read_bytes()).hexdigest())
    with (args.base / 'native_CPU_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
