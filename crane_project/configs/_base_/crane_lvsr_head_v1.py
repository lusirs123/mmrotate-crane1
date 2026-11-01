"""Shared LVSR size-delivery, supervision, training and VAL-gate settings.
The concrete head and branch capacity are selected by the inheriting entry."""

delivery_settings = {'seed': 1703,
 'in_channels': 256,
 'hidden_channels': 8,
 'roi_size': 9,
 'loss_beta': 0.1,
 'edge_ratio_min': 0.8,
 'edge_ratio_max': 1.25,
 'direction_atol_rad': 1e-06}

training = {'epochs': 24,
 'batch_size': 8,
 'seed': 1703,
 'lr': 0.001,
 'weight_decay': 0.0,
 'betas': [0.9, 0.999],
 'eps': 1e-08,
 'clip_norm': 10.0,
 'alignment': 'fixed_b',
 'loss': 'unchanged port_size_qalign_v1.size_loss; SmoothL1 beta0.1 log-size residual',
 'initialization': 'fresh zero terminal; no old LVSR or smoke state',
 'TRAIN': 'all naturally present M outputs at scales0.5/1; no GT-center eligibility filtering',
 'VAL': 'all887 frames native scale; no TEST/depth',
 'continuation': False}

gate = {'AP75_tolerance_pp': 0.0001,
 'size_MAE_tolerance': 1e-06,
 'RIoU_tolerance': 1e-05,
 'AP75_reference': 'max(frozen M20, frozen R4) per overall/real/sim',
 'size_reference': 'frozen M20',
 'joint': 'overall strict gain; per domain nondecrease',
 'protection': 'per-domain AP75,TP50,mean RIoU and exact center/angle/score/presence',
 'ranking': 'overall joint count descending; real long/short then sim long/short MAE ascending; '
            'epoch',
 'no_eligible': 'selected=null; descriptive representative only, never automatic promotion'}

head = {'alignment': 'fixed_b'}

