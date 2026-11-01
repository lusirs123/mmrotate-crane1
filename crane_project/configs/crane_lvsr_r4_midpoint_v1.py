"""R4/M20 pair-specific LVSR entry. L5 remains a VAL descriptive representative."""

_base_ = ['./_base_/crane_lvsr_head_v1.py']

head = {'type': 'PairSpecificSizeHead', 'parameters': 2480}

branches = {'count': 2,
 'input_channels': 267,
 'hidden_channels': 4,
 'output_channels': 4,
 'shared_parameters': False}

training = {'parameters': 2480, 'architecture': 'PairSpecificSizeHead'}

checkpoint = {'epoch': 5,
 'sha256': 'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e',
 'selected_epoch': None,
 'selection_basis': 'original_VAL_descriptive_representative'}


