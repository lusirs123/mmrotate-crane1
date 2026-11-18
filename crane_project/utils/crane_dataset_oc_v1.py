"""Independent native-oc view of the frozen polygons; no annotation writes."""
from pathlib import Path
import numpy as np
from mmrotate.core import poly2obb_np
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.crane_custom_dota import CraneDataset
from mmrotate.datasets.dota import DOTADataset


@ROTATED_DATASETS.register_module()
class CraneDatasetOCV1(CraneDataset):
    def load_annotations(self, ann_folder):
        if self.version != 'oc':
            raise ValueError('This adapter requires the native oc convention')
        infos = super().load_annotations(ann_folder)
        for info in infos:
            boxes, labels = [], []
            path = Path(ann_folder)/(info['img_id']+'.txt')
            for line in path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 9 or parts[8] not in self.CLASSES:
                    continue
                box = poly2obb_np(np.asarray(parts[:8], dtype=np.float32), version='oc')
                if box is None:
                    raise ValueError('Valid frozen polygon failed oc conversion: '+str(path))
                boxes.append(box); labels.append(self.CLASSES.index(parts[8]))
            if labels != info['ann']['labels'].tolist():
                raise ValueError('Changing convention must not change object identities')
            info['ann']['bboxes'] = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        return infos

    def evaluate(self, results, metric='mAP', logger=None, **kwargs):
        # Native online mAP only. Common canonical geometry and temporal
        # metrics are calculated offline, after export, using the sealed GT.
        return DOTADataset.evaluate(self, results, metric=metric, logger=logger, **kwargs)
