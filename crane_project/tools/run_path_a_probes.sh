#!/bin/bash
# =========================================================
# 路径 A 前置诊断：5 个 probe 全面对比
#
# 用途：在投入 Path A（门控+强调制）训练实验前，用现有
#       checkpoint 获取三个可比数据点，判断 injection 是否
#       真正改变 hard-slice 几何。
#
# 前提：
#   - 在项目根目录运行（cd 到 symEOOD/）
#   - GPU 可用（默认 gpu=0，可通过环境变量 GPU 覆盖）
#   - baseline checkpoint 在 work_dirs/crane_symeood_k1/
#   - injector checkpoint 在 work_dirs/crane_symeood_k1_platform_injector/
#   - strong checkpoint 在 work_dirs/crane_symeood_k1_platform_injector_strong_from_k1/
#
# 用法：
#   GPU=0 bash crane_project/tools/run_path_a_probes.sh
#   GPU=0 bash crane_project/tools/run_path_a_probes.sh --skip-strong  # 只跑 P1-P3
# =========================================================

set -e

# ---- 配置 ----
GPU=${GPU:-0}
SEQ="real_seq02"
START=137
END=169
TOPK=50
OUT_DIR="work_dirs/path_a_probes"
SKIP_STRONG=false

if [[ "$1" == "--skip-strong" ]]; then
    SKIP_STRONG=true
fi

# ---- checkpoint 路径 ----
CKPT_BASELINE="work_dirs/crane_symeood_k1/epoch_24.pth"
CKPT_ORDINARY="work_dirs/crane_symeood_k1_platform_injector/epoch_24.pth"
CKPT_STRONG="work_dirs/crane_symeood_k1_platform_injector_strong_from_k1/epoch_20.pth"

# ---- config 路径 ----
CFG_BASELINE="crane_project/configs/crane_symeood_k1.py"
CFG_ORDINARY="crane_project/configs/crane_symeood_k1_platform_injector.py"
CFG_STRONG="crane_project/configs/crane_symeood_k1_platform_injector_strong.py"

# ---- 公共参数 ----
COMMON_ARGS="--split test --seq ${SEQ} --start ${START} --end ${END} \
    --candidate-source main --gpu ${GPU} --topk ${TOPK} \
    --entry-iou-thr 0.10 --usable-iou-thr 0.50"

mkdir -p "${OUT_DIR}"

echo "========================================================"
echo "路径 A 前置诊断"
echo "========================================================"
echo "帧范围:    ${SEQ} [${START}..${END}] ($((END - START + 1)) frames)"
echo "GPU:       ${GPU}"
echo "输出目录:  ${OUT_DIR}"
echo "========================================================"
echo ""

# ---- P1: baseline (无 injector) ----
echo "========== P1: baseline (symeood_k1, 无 injector) =========="
if [[ ! -f "${CKPT_BASELINE}" ]]; then
    echo "[skip] checkpoint 不存在: ${CKPT_BASELINE}"
else
    PYTHONPATH=. python3 crane_project/tools/ctx_entry_probe.py \
        --config "${CFG_BASELINE}" \
        --checkpoint "${CKPT_BASELINE}" \
        ${COMMON_ARGS} \
        --out-json "${OUT_DIR}/P1_baseline.json"
    echo ""
fi

# ---- P2: ordinary injector, 不 inject (当前 probe 行为 = mismatch) ----
echo "========== P2: ordinary injector, NO injection (mismatch) =========="
if [[ ! -f "${CKPT_ORDINARY}" ]]; then
    echo "[skip] checkpoint 不存在: ${CKPT_ORDINARY}"
else
    PYTHONPATH=. python3 crane_project/tools/ctx_entry_probe.py \
        --config "${CFG_ORDINARY}" \
        --checkpoint "${CKPT_ORDINARY}" \
        ${COMMON_ARGS} \
        --out-json "${OUT_DIR}/P2_ordinary_no_injection.json"
    echo ""
fi

# ---- P3: ordinary injector, WITH injection (真实推理行为) ----
echo "========== P3: ordinary injector, WITH injection (真实推理) =========="
if [[ ! -f "${CKPT_ORDINARY}" ]]; then
    echo "[skip] checkpoint 不存在: ${CKPT_ORDINARY}"
else
    PYTHONPATH=. python3 crane_project/tools/ctx_entry_probe.py \
        --config "${CFG_ORDINARY}" \
        --checkpoint "${CKPT_ORDINARY}" \
        ${COMMON_ARGS} \
        --apply-injection \
        --out-json "${OUT_DIR}/P3_ordinary_with_injection.json"
    echo ""
fi

# ---- P4: strong injector, 不 inject ----
if [[ "${SKIP_STRONG}" != "true" ]]; then
    echo "========== P4: strong injector, NO injection (mismatch) =========="
    if [[ ! -f "${CKPT_STRONG}" ]]; then
        echo "[skip] checkpoint 不存在: ${CKPT_STRONG}"
    else
        PYTHONPATH=. python3 crane_project/tools/ctx_entry_probe.py \
            --config "${CFG_STRONG}" \
            --checkpoint "${CKPT_STRONG}" \
            ${COMMON_ARGS} \
            --out-json "${OUT_DIR}/P4_strong_no_injection.json"
        echo ""
    fi

    # ---- P5: strong injector, WITH injection ----
    echo "========== P5: strong injector, WITH injection (真实推理) =========="
    if [[ ! -f "${CKPT_STRONG}" ]]; then
        echo "[skip] checkpoint 不存在: ${CKPT_STRONG}"
    else
        PYTHONPATH=. python3 crane_project/tools/ctx_entry_probe.py \
            --config "${CFG_STRONG}" \
            --checkpoint "${CKPT_STRONG}" \
            ${COMMON_ARGS} \
            --apply-injection \
            --out-json "${OUT_DIR}/P5_strong_with_injection.json"
        echo ""
    fi
fi

# ---- 汇总比较 ----
echo "========================================================"
echo "汇总比较"
echo "========================================================"
PYTHONPATH=. python3 crane_project/tools/compare_path_a_probes.py \
    --out-dir "${OUT_DIR}"
