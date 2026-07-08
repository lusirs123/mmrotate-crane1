#!/bin/bash
# Score-Level Context Modulation 训练脚本
# 基于 k1_brightaug 的 brightness augmentation pipeline
# Context head: BCE+Dice 独立训练
# Score modulation: cls logit + gate_alpha * sigmoid(context_logit.detach())
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="crane_project/configs/crane_symeood_k1_auxctx.py"
GPUS=2
PORT=${PORT:-29500}

# 从 k1 baseline (epoch_24) warm-start，让 backbone/FPN 几何保持
LOAD_FROM="work_dirs/crane_symeood_k1/epoch_24.pth"

echo "=== Score-Level Context Modulation Training ==="
echo "Config:  ${CONFIG}"
echo "GPUs:    ${GPUS}"
echo "Seed:    k1_epoch_24"
echo "Load:    ${LOAD_FROM}"
echo ""

bash tools/dist_train.sh \
    "${CONFIG}" \
    "${GPUS}" \
    --work-dir work_dirs/crane_symeood_k1_auxctx \
    --load-from "${LOAD_FROM}" \
    ${@}
