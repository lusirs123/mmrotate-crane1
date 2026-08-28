"""Source-only trainer for the shared DINO-conditioned geometry refiner."""

import copy
import hashlib
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings

from mmrotate.core import rbbox2result
from mmrotate.models.builder import (ROTATED_DETECTORS, build_detector,
                                     build_head)
from .base import RotatedBaseDetector
from ..roi_heads.dino_conditioned_geometry_refiner import (
    map_model_obb_to_original)


@ROTATED_DETECTORS.register_module()
class SymEOODDinoGeometryRefinerTrainer(RotatedBaseDetector):
    """Train only the refiner from cached DINO OBBs and frozen SymEOOD FPN."""

    def __init__(self,
                 baseline_config,
                 baseline_checkpoint,
                 geometry_refiner,
                 evidence_contract,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        del pretrained
        super().__init__(init_cfg=init_cfg)
        contract = dict(evidence_contract or {})
        required = dict(
            source_train_frames=2781,
            source_val_frames=738,
            target_data_read=False,
            detector_forward_during_training=False,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False)
        for key, expected in required.items():
            if contract.get(key) != expected:
                raise ValueError(
                    'Geometry-refiner evidence contract mismatch for {}: '
                    'expected {!r}, got {!r}'.format(
                        key, expected, contract.get(key)))
        self.evidence_contract = contract
        baseline_path = os.path.abspath(os.fspath(baseline_config))
        baseline_cfg = Config.fromfile(baseline_path)
        imports = baseline_cfg.get('custom_imports')
        if imports:
            import_modules_from_strings(**imports)
        baseline_model_cfg = copy.deepcopy(baseline_cfg.model)
        baseline_model_cfg.pretrained = None
        self.baseline = build_detector(baseline_model_cfg)
        checkpoint_path = os.path.abspath(os.fspath(baseline_checkpoint))
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                'Frozen SymEOOD checkpoint does not exist: '
                + checkpoint_path)
        load_checkpoint(
            self.baseline, checkpoint_path, map_location='cpu', strict=False)
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.baseline.eval()
        self.geometry_refiner = build_head(dict(geometry_refiner))
        self.CLASSES = getattr(self.baseline, 'CLASSES', ('grab',))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._frozen_hash_at_init = self.frozen_parameter_hash()

    def train(self, mode=True):
        super().train(mode)
        # MMCV runners call model.train() every epoch.  Keep BN/running state
        # and every frozen SymEOOD module in inference mode unconditionally.
        self.baseline.eval()
        self.geometry_refiner.train(mode)
        return self

    def extract_feat(self, img):
        with torch.no_grad():
            return tuple(feature.detach()
                         for feature in self.baseline.extract_feat(img))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      dino_proposals,
                      gt_bboxes_ignore=None,
                      **kwargs):
        del gt_labels, gt_bboxes_ignore, kwargs
        super().forward_train(img, img_metas)
        if len(dino_proposals) != len(gt_bboxes):
            raise RuntimeError('DINO/GT batch-size mismatch')
        features = self.extract_feat(img)
        proposal_list = [item[:, :5].to(features[0].device)
                         for item in dino_proposals]
        gt_box_list = [item[:, :5].to(features[0].device)
                       for item in gt_bboxes]
        predicted = self.geometry_refiner(features, proposal_list)
        targets = self.geometry_refiner.encode_targets(
            proposal_list, gt_box_list)
        losses = self.geometry_refiner.loss(predicted, targets)
        losses['refiner_sample_count'] = predicted.new_tensor(
            float(predicted.shape[0]))
        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    dino_proposals,
                    rescale=False,
                    **kwargs):
        del kwargs
        features = self.extract_feat(img)
        proposal_list = [item[:, :5].to(features[0].device)
                         for item in dino_proposals]
        predicted = self.geometry_refiner(features, proposal_list)
        decoded = self.geometry_refiner.decode_and_normalize(
            proposal_list, predicted, img_metas=img_metas)
        outputs = []
        for index, boxes in enumerate(decoded):
            if boxes.shape[0] == 0:
                image_features = tuple(
                    feature[index:index + 1] for feature in features)
                fallback = self.baseline.simple_test_from_features(
                    image_features, [img_metas[index]],
                    rescale=rescale)[0]
                outputs.append(fallback)
                continue
            if rescale:
                boxes = map_model_obb_to_original(boxes, img_metas[index])
            detections = torch.cat(
                [boxes, boxes.new_ones((boxes.shape[0], 1))], dim=1)
            labels = torch.zeros(
                boxes.shape[0], dtype=torch.long, device=boxes.device)
            outputs.append(rbbox2result(detections, labels, 1))
        return outputs

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        del imgs, img_metas, rescale, kwargs
        raise RuntimeError(
            'Geometry-refiner trainer does not support test-time augmentation')

    def frozen_parameter_hash(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.baseline.state_dict().items()):
            digest.update(name.encode('utf-8'))
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(np.asarray(array).tobytes())
        return digest.hexdigest()

    def verify_frozen_contract(self):
        current = self.frozen_parameter_hash()
        return dict(
            baseline_eval=not self.baseline.training,
            baseline_trainable_parameter_count=sum(
                int(parameter.numel()) for parameter in
                self.baseline.parameters() if parameter.requires_grad),
            refiner_trainable_parameter_count=sum(
                int(parameter.numel()) for parameter in
                self.geometry_refiner.parameters()
                if parameter.requires_grad),
            frozen_hash_at_init=self._frozen_hash_at_init,
            frozen_hash_current=current,
            frozen_hash_unchanged=(current == self._frozen_hash_at_init),
            refiner_contract=self.geometry_refiner.component_contract(),
            evidence_contract=dict(self.evidence_contract))
