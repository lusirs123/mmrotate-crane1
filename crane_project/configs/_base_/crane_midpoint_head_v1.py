"""Shared four-midpoint head, local sampling, decoder guards and formal training recipe."""

head = {'type': 'SigmaMidpointHead', 'prior_sigma_cells': 1.0, 'parameters': 17696}

implementation_settings = {'seed': 1703,
 'permutation_seed': 1704,
 'in_channels': 256,
 'hidden_channels': 32,
 'roi_size': 9,
 'context_multiplier': 1.5,
 'stride': 8,
 'min_context_cells': 2.0,
 'prior_sigma_cells': 1.0,
 'loss_beta': 0.1,
 'steps_per_arm': 200,
 'batch_size': 8,
 'lr': 0.001,
 'weight_decay': 0.0,
 'clip_norm': 10.0,
 'max_center_over_b_short': 0.3,
 'max_log_edge_residual': 0.22314355131420976,
 'max_angle_residual_rad': 0.17453292519943295,
 'min_pair_length_px': 1e-06,
 'roi_bound_tolerance': 1e-05}

sampling = {'feature_level': 0,
 'stride': 8,
 'in_channels': 256,
 'roi_size': 9,
 'context_multiplier': 1.5,
 'min_context_cells': 2.0}

training = {'epochs': 24,
 'batch_size': 8,
 'seed': 1703,
 'lr': 0.001,
 'weight_decay': 0.0,
 'clip_norm': 10.0,
 'train_scales': [1.0, 0.5],
 'val_scale': 1.0,
 'max_cpu_tensor_bytes': 1073741824,
 'train_center_match_px': 15.0,
 'balanced_batch_real': 4,
 'balanced_batch_sim': 4}

