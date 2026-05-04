# mmrotate/datasets/crane_custom_dota.py
import os.path as osp
import numpy as np
import glob
import os 
import re
from mmrotate.core import poly2obb_np
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
from mmcv.utils import print_log

def _parse_domain(filename):
    """从文件名提取域标签：real_xxx -> 'real', sim_xxx -> 'sim'"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^(real|sim)_', basename)
    return m.group(1) if m else 'unknown'

@ROTATED_DATASETS.register_module(force=True)
class CraneDataset(DOTADataset):
    CLASSES = ('grab',)

    def load_annotations(self, ann_folder):
        ann_files = glob.glob(os.path.join(ann_folder, '*.txt'))
        data_infos = []

        for ann_file in sorted(ann_files):
            bboxes, labels = [], []

            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    cls_name = parts[8]
                    if cls_name not in self.CLASSES:
                        continue
                    poly = np.array([float(x) for x in parts[:8]], dtype=np.float32)
                    obb = poly2obb_np(poly, version='le90')
                    if obb is None:
                        continue
                    bboxes.append(obb)
                    labels.append(self.CLASSES.index(cls_name))

            img_name = os.path.splitext(os.path.basename(ann_file))[0] + '.jpg'
            domain = _parse_domain(ann_file)
            img_id = os.path.splitext(os.path.basename(ann_file))[0]  # ★ 新增

            bboxes_arr = np.array(bboxes, dtype=np.float32) if bboxes \
                else np.zeros((0, 5), dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64) if labels \
                else np.zeros((0,), dtype=np.int64)

            data_infos.append(dict(
                filename=img_name,
                domain=domain,
                img_id=img_id,           # ★ 新增：format_results 需要此字段
                ann=dict(
                    bboxes=bboxes_arr,
                    labels=labels_arr,
                    bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                    labels_ignore=np.zeros((0,), dtype=np.int64),
                )
            ))

        # ★ 新增：设置 img_ids 属性，供 format_results/merge_det 使用
        self.img_ids = [info['img_id'] for info in data_infos]
        
        print(f'[CraneDataset] 成功加载 {len(data_infos)} 条标注，来源：{ann_folder}')
        return data_infos
    
    def evaluate(self, results, metric='mAP', logger=None, **kwargs):
        thresh_sim  = kwargs.pop('thresh_sim',  10.0)
        thresh_real = kwargs.pop('thresh_real', 25.0)
        weight_sim  = kwargs.pop('weight_sim',  0.7)
        weight_real = kwargs.pop('weight_real', 0.3)

        eval_results = {}
        if metric == 'mAP':
            eval_results.update(super().evaluate(results, metric, logger, **kwargs))

        hits_sim, hits_real = [], []

        for i, result in enumerate(results):
            info   = self.data_infos[i]
            domain = info.get('domain', 'unknown')

            ann       = self.get_ann_info(i)
            gt_bboxes = ann['bboxes']          # shape (N, 5)：cx,cy,w,h,theta
            if gt_bboxes.shape[0] == 0:
                continue

            # ★ 病灶二修复：OBB格式直接取前两列作为质心
            gt_center = gt_bboxes[0, :2]       # [cx, cy]

            pred_bboxes = result[0]            # (K, 6)：cx,cy,w,h,theta,score
            if pred_bboxes.shape[0] > 0:
                # 取得分最高的预测框（已按score降序排列）
                pred_center = pred_bboxes[0, :2]   # [cx, cy]
                dist = np.linalg.norm(pred_center - gt_center)

                if domain == 'sim':
                    hits_sim.append(float(dist < thresh_sim))
                elif domain == 'real':
                    hits_real.append(float(dist < thresh_real))
            else:
                if domain == 'sim':
                    hits_sim.append(0.0)
                elif domain == 'real':
                    hits_real.append(0.0)

        r_center_sim  = float(np.mean(hits_sim))  if hits_sim  else 0.0
        r_center_real = float(np.mean(hits_real)) if hits_real else 0.0
        weighted_score = weight_sim * r_center_sim + weight_real * r_center_real

        eval_results['sim_R_center']      = r_center_sim
        eval_results['real_R_center']     = r_center_real
        eval_results['Weighted_R_center'] = weighted_score

        print_log(f'\n[SIM 域绝对精度] R_center (<{thresh_sim}px): {r_center_sim:.4f}',   logger=logger)
        print_log(f'[REAL 域泛化哨兵] R_center (<{thresh_real}px): {r_center_real:.4f}',  logger=logger)
        print_log(f'[全局加权决策分] Weighted_R_center: {weighted_score:.4f}',             logger=logger)
        
        # 诊断：打印域分布统计
        print_log(f'[域统计] sim帧={len(hits_sim)}, real帧={len(hits_real)}, unknown帧跳过', logger=logger)

        return eval_results

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """绕过 DOTA patch merge 逻辑，直接输出单图预测结果"""
        import os
        import re
        assert isinstance(results, list), 'results 必须是 list'
        
        if submission_dir is None:
            submission_dir = '/tmp/crane_preds'
        
        save_dir = os.path.join(submission_dir, 'Task1_grab')
        os.makedirs(save_dir, exist_ok=True)

        for idx, result in enumerate(results):
            img_id = self.data_infos[idx]['img_id']
            out_path = os.path.join(save_dir, f'{img_id}.txt')
            
            pred_bboxes = result[0]  # class 0 (grab) 的预测框，shape (K, 6)
            
            with open(out_path, 'w') as f:
                if pred_bboxes.shape[0] == 0:
                    pass  # 空文件，表示无检测结果
                else:
                    for bbox in pred_bboxes:
                        # bbox: [cx, cy, w, h, theta, score]
                        cx, cy, w, h, theta, score = bbox
                        # 转回多边形格式（DOTA 标准输出）
                        import cv2
                        import numpy as np
                        rect = ((float(cx), float(cy)), 
                                (float(w), float(h)), 
                                float(np.degrees(theta)))
                        pts = cv2.boxPoints(rect)  # (4, 2)
                        pts = pts.flatten()
                        coords = ' '.join([f'{p:.2f}' for p in pts])
                        f.write(f'{coords} {float(score):.4f}\n')
        
        print(f'[CraneDataset] 预测结果已写入 {save_dir}，共 {len(results)} 帧')
        return None, save_dir