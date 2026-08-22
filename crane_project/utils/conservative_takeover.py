"""Source-calibrated conservative SymEOOD/DINO lane arbitration.

The selector is deliberately model-agnostic and contains no target-specific
rules.  It applies an asymmetric score margin, causal confirmation, and a
bounded geometry-change gate while preserving the geometry owned by the
selected proposal lane.
"""

import math
from typing import Dict

import numpy as np


def _box_array(box):
    if box is None:
        return None
    array = np.asarray(box, dtype=np.float32).reshape(-1)
    if array.size < 6 or not np.isfinite(array[:6]).all():
        return None
    if float(array[2]) <= 0.0 or float(array[3]) <= 0.0:
        return None
    return array[:6].copy()


def geometry_change(previous, current) -> Dict[str, float]:
    """Return scale-normalized, pi-periodic change between two OBBs."""
    previous = _box_array(previous)
    current = _box_array(current)
    if previous is None or current is None:
        return dict(diag_change=float('inf'), angle_change_deg=float('inf'))
    previous_diag = math.hypot(float(previous[2]), float(previous[3]))
    current_diag = math.hypot(float(current[2]), float(current[3]))
    diag_change = abs(current_diag - previous_diag) / max(previous_diag, 1e-6)
    delta = float(current[4] - previous[4])
    delta = (delta + math.pi / 2.0) % math.pi - math.pi / 2.0
    return dict(
        diag_change=float(diag_change),
        angle_change_deg=float(abs(math.degrees(delta))))


class ConservativeTakeoverSelector:
    """Causal lane-level hysteresis with a SymEOOD-safe default."""

    def __init__(self, enter_margin, exit_margin, min_confirmations,
                 max_diag_change, max_angle_change_deg):
        self.enter_margin = float(enter_margin)
        self.exit_margin = float(exit_margin)
        self.min_confirmations = int(min_confirmations)
        self.max_diag_change = float(max_diag_change)
        self.max_angle_change_deg = float(max_angle_change_deg)
        if self.enter_margin < 0.0:
            raise ValueError('enter_margin must be non-negative')
        if self.exit_margin > self.enter_margin:
            raise ValueError('exit_margin must not exceed enter_margin')
        if self.min_confirmations < 1:
            raise ValueError('min_confirmations must be positive')
        if self.max_diag_change <= 0.0 or self.max_angle_change_deg <= 0.0:
            raise ValueError('geometry limits must be positive')
        self.reset()

    def reset(self):
        self.previous_sequence = None
        self.previous_frame = None
        self.previous_box = None
        self.active_source = 'sym_eood'
        self.pending_dino = 0

    def _continuous(self, sequence, frame):
        return (self.previous_sequence == str(sequence)
                and self.previous_frame is not None
                and int(frame) == int(self.previous_frame) + 1)

    def _geometry_allowed(self, dino_box):
        if self.previous_box is None:
            return True, dict(diag_change=0.0, angle_change_deg=0.0)
        change = geometry_change(self.previous_box, dino_box)
        allowed = (change['diag_change'] <= self.max_diag_change
                   and change['angle_change_deg'] <= self.max_angle_change_deg)
        return bool(allowed), change

    def select(self, sym_box, dino_box, sequence, frame):
        sym_box = _box_array(sym_box)
        dino_box = _box_array(dino_box)
        if not self._continuous(sequence, frame):
            self.reset()

        sym_score = None if sym_box is None else float(sym_box[5])
        dino_score = None if dino_box is None else float(dino_box[5])
        score_delta = (None if sym_score is None or dino_score is None
                       else float(dino_score - sym_score))
        geometry_allowed, change = self._geometry_allowed(dino_box)

        if sym_box is None and dino_box is None:
            selected, source, reason = None, 'missing', 'both_lanes_missing'
            self.pending_dino = 0
        elif sym_box is None:
            selected, source, reason = (
                dino_box, 'dino_native', 'sym_eood_missing_immediate_rescue')
            self.pending_dino = 0
        elif dino_box is None:
            selected, source, reason = (
                sym_box, 'sym_eood', 'dino_missing_keep_sym_eood')
            self.pending_dino = 0
        elif self.active_source == 'dino_native':
            if not geometry_allowed:
                selected, source, reason = (
                    sym_box, 'sym_eood', 'dino_geometry_rejected_exit')
                self.pending_dino = 0
            elif score_delta >= self.exit_margin:
                selected, source, reason = (
                    dino_box, 'dino_native', 'dino_hysteresis_hold')
                self.pending_dino = 0
            else:
                selected, source, reason = (
                    sym_box, 'sym_eood', 'score_exit_to_sym_eood')
                self.pending_dino = 0
        elif score_delta >= self.enter_margin and geometry_allowed:
            self.pending_dino += 1
            if self.pending_dino >= self.min_confirmations:
                selected, source, reason = (
                    dino_box, 'dino_native', 'confirmed_score_takeover')
                self.pending_dino = 0
            else:
                selected, source, reason = (
                    sym_box, 'sym_eood', 'awaiting_dino_confirmation')
        else:
            selected, source = sym_box, 'sym_eood'
            reason = ('dino_geometry_rejected' if not geometry_allowed
                      else 'dino_margin_insufficient')
            self.pending_dino = 0

        previous_source = self.active_source
        if source in ('sym_eood', 'dino_native'):
            self.active_source = source
        self.previous_sequence = str(sequence)
        self.previous_frame = int(frame)
        self.previous_box = None if selected is None else selected[:6].copy()
        return dict(
            selected=selected,
            selected_source=source,
            previous_source=previous_source,
            source_switched=bool(source != previous_source),
            takeover_reason=reason,
            score_delta=score_delta,
            geometry_allowed=geometry_allowed,
            diag_change=change['diag_change'],
            angle_change_deg=change['angle_change_deg'])
