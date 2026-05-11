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
    
    # 新增 sim/A—RMSE 绝对角度精度作为早停指标
    def evaluate(self, results, metric='mAP', logger=None, **kwargs):
        """
        面向 CPS 闭环的跨域非对称评估主入口。
        核心逻辑：
        1. 空间与姿态的因果约束：仅在质心物理命中（< thresh_sim）时，评估姿态误差。
        2. 流形折叠：利用模运算处理 OBB 旋转角的 180° 等价性。
        3. 极端惩罚：全漏检或全假阳性时，返回 90.0° 的 A-RMSE 最大理论盲区值。
        """
        from mmcv.utils import print_log
        import numpy as np

        thresh_sim  = kwargs.pop('thresh_sim',  10.0)
        thresh_real = kwargs.pop('thresh_real', 25.0)
        weight_sim  = kwargs.pop('weight_sim',  0.6)
        weight_real = kwargs.pop('weight_real', 0.4)

        eval_results = {}

        # 1. 保留标准 mAP 评估（兼容 COCO 静态基线对照）
        if metric == 'mAP':
            try:
                eval_results.update(super().evaluate(results, metric, logger, **kwargs))
            except Exception as e:
                print_log(f'[CraneDataset] 静态 mAP 计算跳过或异常：{e}', logger=logger)

        # 2. 逐帧物理量测收集
        hits_sim,  hits_real  = [], []
        sim_a_gt,  sim_a_pred = [], []

        for i, result in enumerate(results):
            info   = self.data_infos[i]
            domain = info.get('domain', 'unknown')

            ann       = self.get_ann_info(i)
            gt_bboxes = ann['bboxes']          
            if gt_bboxes.shape[0] == 0:        # 负样本帧过滤
                continue

            gt_center = gt_bboxes[0, :2]
            gt_angle  = float(gt_bboxes[0, 4])

            pred_bboxes = result[0]            # 类别 0 (grab) 预测输出

            if pred_bboxes.shape[0] > 0:
                pred_center = pred_bboxes[0, :2]
                pred_angle  = float(pred_bboxes[0, 4])
                dist        = float(np.linalg.norm(pred_center - gt_center))

                if domain == 'sim':
                    is_hit = dist < thresh_sim
                    hits_sim.append(float(is_hit))
                    # [绝对因果约束]：仅空间命中，姿态收集方有物理意义
                    if is_hit:
                        sim_a_gt.append(gt_angle)
                        sim_a_pred.append(pred_angle)
                elif domain == 'real':
                    hits_real.append(float(dist < thresh_real))
            else:
                # 漏检记录
                if domain == 'sim':
                    hits_sim.append(0.0)
                elif domain == 'real':
                    hits_real.append(0.0)

        # 3. 跨域空间召回率计算
        r_center_sim  = float(np.mean(hits_sim))  if hits_sim  else 0.0
        r_center_real = float(np.mean(hits_real)) if hits_real else 0.0
        weighted_score = weight_sim * r_center_sim + weight_real * r_center_real

        # 4. 孪生域绝对角度精度 A-RMSE 计算
        if len(sim_a_gt) > 0:
            diff = np.array(sim_a_pred) - np.array(sim_a_gt)
            # [流形折叠]：消除 OBB 周期跳变假象，映射至 [-pi/2, pi/2)
            diff = (diff + np.pi / 2) % np.pi - np.pi / 2
            sim_a_rmse = float(np.degrees(np.sqrt(np.mean(diff ** 2))))
        else:
            sim_a_rmse = 90.0  # [重罚机制]：无任何有效空间命中，触发最高姿态误差

        # 5. 注入框架验证字典
        eval_results['sim_A_RMSE']        = round(sim_a_rmse,    4)
        eval_results['sim_R_center']      = round(r_center_sim,  4)
        eval_results['real_R_center']     = round(r_center_real, 4)
        eval_results['Weighted_R_center'] = round(weighted_score, 4)

        # 6. 终端日志规整化输出
        sep = '=' * 60
        print_log(
            f'\n{sep}\n'
            f'  [面向控制的视觉评价结果 (Test Mode)]\n'
            f'{sep}\n'
            f'  [SIM 域]  时序姿态本底 A-RMSE  : {sim_a_rmse:.2f}° '
            f'({len(sim_a_gt)}/{len(hits_sim)} 帧质心合法)\n'
            f'  [SIM 域]  静态质心容差 <{thresh_sim:.0f}px  : {r_center_sim:.4f}  ({len(hits_sim)} 帧)\n'
            f'  [REAL 域] 极点泛化容差 <{thresh_real:.0f}px  : {r_center_real:.4f}  ({len(hits_real)} 帧)\n'
            f'  [综合决策] 加权保存基准 (W_R_c) : {weighted_score:.4f}\n'
            f'{sep}',
            logger=logger
        )

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