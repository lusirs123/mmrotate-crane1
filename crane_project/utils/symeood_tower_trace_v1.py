"""Preserve reference TRAIN RNG/input stream for the added-tower control."""
import base64
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS
from crane_project.utils.symeood_repro_trace_v1 import SymEOODReproTraceV1


@HOOKS.register_module()
class SymEOODTowerTraceV1(SymEOODReproTraceV1):
    def __init__(self, reference_work, **kwargs):
        super().__init__(**kwargs)
        self.reference_work = Path(reference_work)

    def before_run(self, runner):
        p = self.reference_work / ('repro_initial_rank%d.jsonl' % dist.get_rank())
        rows = p.read_text().splitlines()
        assert len(rows) == 1
        # Trusted, existing local run record; never an external pickle source.
        state = pickle.loads(base64.b64decode(json.loads(rows[0])['rng']))
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'])
        super().before_run(runner)
