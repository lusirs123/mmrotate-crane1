"""PQA v3: inference-aligned decoded-candidate ranking supervision.

V2 fits a dense Gaussian heatmap but inference takes a quality-only maximum
over 10,000 decoded candidates.  On unseen real dark frames, one false PQA
peak can therefore win even when the average heatmap loss is small.  V3 keeps
the detached PQA architecture and main BrightAug detector unchanged, and adds
pairwise ordering supervision on the actual decoded cls-topK candidates.

Unified protocol:
  * initialize from K1 epoch_24 through command-line load_from;
  * train a fresh 24-epoch branch with root tools/dist_train.sh (seed=0);
  * scan only epochs 16/18/20/22/24.
"""

_base_ = ['./crane_symeood_k1_pqa_quality_primary_v2.py']

model = dict(
    # At an uninformative score tie the smooth rank loss is about 0.1.  These
    # weights put clean/dark rank terms near 0.01/0.005, comparable to the
    # observed V2 dense LD loss without overwhelming spatial supervision.
    pqa_rank_loss_weight=0.10,
    pqa_dark_rank_loss_weight=0.05,
    pqa_rank_samples=128,
    # Coarse full-pool mining catches current PQA false maxima cheaply; the
    # selected 128 candidates still use the inference 25x25 integration.
    pqa_rank_mining_grid_size=5,
    pqa_rank_min_iou_gap=0.10,
    pqa_rank_score_margin=0.05,
    pqa_rank_temperature=0.10)

load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_pqa_quality_primary_v3_rank'
