#!/usr/bin/env bash
# =========================================================
# 实验 D: QFL (Quality Focal Loss) 训练脚本
#
# 从 k1 baseline epoch_24 warm start
# 核心改动：loss_cls 中 use_quality_target=True
# =========================================================

set -euo pipefail

CONFIG="crane_project/configs/crane_symeood_k1_qfl.py"
GPUS=2
LOAD_FROM="work_dirs/crane_symeood_k1/epoch_24.pth"

echo "========================================================"
echo "  QFL Training"
echo "  Config:  ${CONFIG}"
echo "  GPUs:    ${GPUS}"
echo "  Load:    ${LOAD_FROM}"
echo "========================================================"

bash crane_project/tools/dist_train.sh \
    "${CONFIG}" \
    ${GPUS} \
    --cfg-options load_from="${LOAD_FROM}"
