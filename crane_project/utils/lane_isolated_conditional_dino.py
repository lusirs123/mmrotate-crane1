"""Lane-isolated conditional DINO rescue for sequential OBB inference.

The policy deliberately decides whether to invoke DINO from SymEOOD geometry
and SymEOOD's own causal history only.  Once DINO is invoked, its geometry is
checked against the previous *DINO* observation, never against the previously
selected lane.  A DINO geometry discontinuity marks the measurement as risky
but does not force a switch to a potentially wrong SymEOOD box.
"""

import math
from typing import Dict, Iterable, Tuple

import numpy as np

from crane_project.utils.conservative_takeover import (
    _box_array, geometry_change)


def normalized_diagonal(box, image_shape: Iterable[int]) -> float:
    """Return OBB diagonal divided by image diagonal."""
    box = _box_array(box)
    shape = tuple(int(value) for value in image_shape)
    if box is None or len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        return 0.0
    box_diag = math.hypot(float(box[2]), float(box[3]))
    image_diag = math.hypot(float(shape[0]), float(shape[1]))
    return float(box_diag / max(image_diag, 1e-6))


def _bounded_change(previous, current) -> Dict[str, float]:
    if previous is None or current is None:
        return dict(diag_change=0.0, angle_change_deg=0.0)
    return geometry_change(previous, current)


class LaneIsolatedConditionalDinoSelector:
    """Two-phase, causal selector that can skip DINO before its forward pass."""

    def __init__(self, small_diag_ratio, max_sym_diag_change,
                 max_sym_angle_change_deg, max_dino_diag_change,
                 max_dino_angle_change_deg):
        self.small_diag_ratio = float(small_diag_ratio)
        self.max_sym_diag_change = float(max_sym_diag_change)
        self.max_sym_angle_change_deg = float(max_sym_angle_change_deg)
        self.max_dino_diag_change = float(max_dino_diag_change)
        self.max_dino_angle_change_deg = float(max_dino_angle_change_deg)
        if self.small_diag_ratio < 0.0:
            raise ValueError('small_diag_ratio must be non-negative')
        for name, value in (
                ('max_sym_diag_change', self.max_sym_diag_change),
                ('max_sym_angle_change_deg', self.max_sym_angle_change_deg),
                ('max_dino_diag_change', self.max_dino_diag_change),
                ('max_dino_angle_change_deg',
                 self.max_dino_angle_change_deg)):
            if value <= 0.0:
                raise ValueError('{} must be positive'.format(name))
        self.reset()

    def reset(self):
        self.previous_sequence = None
        self.previous_frame = None
        self.previous_sym_box = None
        self.previous_dino_box = None
        self.previous_dino_sequence = None
        self.previous_dino_frame = None
        self._pending = None

    def _continuous(self, sequence, frame) -> bool:
        return (self.previous_sequence == str(sequence)
                and self.previous_frame is not None
                and int(frame) == int(self.previous_frame) + 1)

    def _dino_continuous(self, sequence, frame) -> bool:
        return (self.previous_dino_sequence == str(sequence)
                and self.previous_dino_frame is not None
                and int(frame) == int(self.previous_dino_frame) + 1)

    def begin_frame(self, sym_box, image_shape: Tuple[int, int], sequence,
                    frame) -> Dict:
        """Decide whether DINO is needed without reading any DINO output."""
        if self._pending is not None:
            raise RuntimeError('finish_frame must be called before begin_frame')
        if not self._continuous(sequence, frame):
            self.reset()

        sym_box = _box_array(sym_box)
        sym_change = _bounded_change(self.previous_sym_box, sym_box)
        diag_ratio = normalized_diagonal(sym_box, image_shape)
        reasons = []
        if sym_box is None:
            reasons.append('sym_eood_missing')
        else:
            if diag_ratio <= self.small_diag_ratio:
                reasons.append('sym_eood_small_geometry')
            if self.previous_sym_box is not None:
                if sym_change['diag_change'] > self.max_sym_diag_change:
                    reasons.append('sym_eood_diag_discontinuity')
                if (sym_change['angle_change_deg']
                        > self.max_sym_angle_change_deg):
                    reasons.append('sym_eood_angle_discontinuity')

        self._pending = dict(
            sym_box=sym_box,
            image_shape=tuple(int(value) for value in image_shape[:2]),
            sequence=str(sequence),
            frame=int(frame),
            invoke_dino=bool(reasons),
            trigger_reasons=reasons,
            sym_normalized_diag=float(diag_ratio),
            sym_diag_change=float(sym_change['diag_change']),
            sym_angle_change_deg=float(sym_change['angle_change_deg']))
        return dict(self._pending)

    def finish_frame(self, dino_box=None) -> Dict:
        """Finish the pending decision after an optional DINO forward pass."""
        if self._pending is None:
            raise RuntimeError('begin_frame must be called before finish_frame')
        pending = self._pending
        sym_box = pending['sym_box']
        invoke_dino = bool(pending['invoke_dino'])
        dino_box = _box_array(dino_box) if invoke_dino else None
        dino_change = _bounded_change(
            self.previous_dino_box if self._dino_continuous(
                pending['sequence'], pending['frame']) else None,
            dino_box)
        dino_geometry_stable = bool(
            dino_box is not None
            and dino_change['diag_change'] <= self.max_dino_diag_change
            and dino_change['angle_change_deg']
            <= self.max_dino_angle_change_deg)

        if invoke_dino and dino_box is not None:
            selected = dino_box
            selected_source = 'dino_native'
            reason = 'conditional_dino_rescue'
            measurement_valid = dino_geometry_stable
            risk_reasons = ([] if dino_geometry_stable else
                            ['dino_self_geometry_discontinuity'])
        elif sym_box is not None:
            selected = sym_box
            selected_source = 'sym_eood'
            reason = ('dino_missing_keep_sym_eood' if invoke_dino else
                      'conditional_dino_not_required')
            measurement_valid = not invoke_dino
            risk_reasons = ([] if measurement_valid else
                            ['requested_dino_missing'])
        else:
            selected = None
            selected_source = 'missing'
            reason = 'both_lanes_missing'
            measurement_valid = False
            risk_reasons = ['measurement_missing']

        self.previous_sequence = pending['sequence']
        self.previous_frame = pending['frame']
        self.previous_sym_box = (
            None if sym_box is None else sym_box[:6].copy())
        if invoke_dino:
            self.previous_dino_sequence = pending['sequence']
            self.previous_dino_frame = pending['frame']
            self.previous_dino_box = (
                None if dino_box is None else dino_box[:6].copy())
        self._pending = None

        result = dict(pending)
        result.update(dict(
            selected=selected,
            selected_source=selected_source,
            selection_reason=reason,
            dino_available=bool(dino_box is not None),
            dino_geometry_stable=bool(dino_geometry_stable),
            dino_diag_change=float(dino_change['diag_change']),
            dino_angle_change_deg=float(dino_change['angle_change_deg']),
            measurement_valid=bool(measurement_valid),
            risk_reasons=risk_reasons))
        return result

    def select(self, sym_box, dino_box, image_shape, sequence, frame) -> Dict:
        """Offline convenience wrapper using already collected lane outputs."""
        self.begin_frame(sym_box, image_shape, sequence, frame)
        return self.finish_frame(dino_box)
