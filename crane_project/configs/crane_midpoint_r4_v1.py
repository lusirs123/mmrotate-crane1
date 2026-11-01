"""R4-specific sigma1.5 midpoint entry, fixed at original VAL-selected M20."""

_base_ = ['./_base_/crane_midpoint_head_v1.py']

head = {'prior_sigma_cells': 1.5}

checkpoint = {'epoch': 20,
 'sha256': 'd595ab01c3aea48aec66d6cf1e38460990ea272bb5ca962ca8ec8e17dd8d6dce',
 'selection_basis': 'original_full_VAL_rule'}


