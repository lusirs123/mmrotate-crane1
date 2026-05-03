# mmrotate_crane/__init__.py
# MMRotate 自定义模块包入口
# 通过 custom_imports 机制注册 CraneOBBMetric

# crane_project/mmrotate_crane/__init__.py

from .sym_eood import (SymPOLAAssigner, SymKLDLoss, SymNFLLoss,
                       SymEOODHead, SymEOOD ,sym_kld,CraneOBBMetric,
                       evaluate_from_pkl,)

__all__ = [
    "SymEOOD",
    "sym_kld",
    'CraneOBBMetric',
    'evaluate_from_pkl',
    'SymPOLAAssigner', 
    'SymKLDLoss', 
    'SymNFLLoss', 
    'SymEOODHead'
]
