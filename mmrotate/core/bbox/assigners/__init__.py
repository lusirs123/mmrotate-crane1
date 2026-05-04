# Copyright (c) OpenMMLab. All rights reserved.
from .atss_kld_assigner import ATSSKldAssigner
from .atss_obb_assigner import ATSSObbAssigner
from .convex_assigner import ConvexAssigner
from .max_convex_iou_assigner import MaxConvexIoUAssigner
from .sas_assigner import SASAssigner
from .pola import PolaAssigner


from .sym_pola import SymPOLAAssigner

__all__ = [
    'ConvexAssigner', 'MaxConvexIoUAssigner', 'SASAssigner', 'ATSSKldAssigner',
    'ATSSObbAssigner', 'PolaAssigner',
    'SymPOLAAssigner'
]
