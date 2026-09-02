"""Source-only trainer for the shared DINO-conditioned geometry refiner."""

import copy
import hashlib
import os
import re

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


def _unwrap_single_augmentation_proposals(dino_proposals):
    """Remove the one augmentation dimension created by test pipelines."""
    if not isinstance(dino_proposals, (list, tuple)):
        raise RuntimeError('DINO proposals must be a batch list')
    nested = any(isinstance(item, (list, tuple))
                 for item in dino_proposals)
    if nested:
        if len(dino_proposals) != 1:
            raise RuntimeError(
                'Geometry refiner supports exactly one test augmentation')
        dino_proposals = dino_proposals[0]
    if (not isinstance(dino_proposals, (list, tuple))
            or any(not torch.is_tensor(item) for item in dino_proposals)):
        raise RuntimeError(
            'DINO proposal test structure is not augmentation-by-batch')
    return list(dino_proposals)


def _unwrap_single_augmentation_tensor(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(
                '{} supports exactly one test augmentation'.format(name))
        value = value[0]
    if not torch.is_tensor(value):
        raise RuntimeError('{} must be a tensor'.format(name))
    return value


@ROTATED_DETECTORS.register_module()
class SymEOODDinoGeometryRefinerTrainer(RotatedBaseDetector):
    """Train only the refiner from cached DINO OBBs and frozen SymEOOD FPN."""

    def __init__(self,
                 baseline_config,
                 baseline_checkpoint,
                 geometry_refiner,
                 evidence_contract,
                 geometry_refiner_checkpoint=None,
                 geometry_refiner_checkpoint_sha256=None,
                 geometry_refiner_checkpoint_contract=None,
                 evaluation_only=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        del pretrained
        super().__init__(init_cfg=init_cfg)
        contract = dict(evidence_contract or {})
        frozen_k1_head_forward = bool(
            contract.get('frozen_symeood_detection_head_forward', False))
        source_train_frames = contract.get('source_train_frames')
        if (not isinstance(source_train_frames, int)
                or isinstance(source_train_frames, bool)
                or source_train_frames <= 0):
            raise ValueError(
                'Geometry-refiner source_train_frames must be positive')
        required = dict(
            source_val_frames=738,
            target_data_read=False,
            detector_forward_during_training=frozen_k1_head_forward,
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
        self.evaluation_only = bool(evaluation_only)
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
        refiner_contract = self.geometry_refiner.component_contract()
        self.uses_k1_geometry_anchor = bool(
            refiner_contract.get('current_k1_geometry_anchor', False))
        self.uses_symmetric_dual_candidate_anchor = bool(
            refiner_contract.get('symmetric_dual_candidate_anchor', False))
        self.geometry_refiner_initialization = self._load_refiner_checkpoint(
            geometry_refiner_checkpoint,
            geometry_refiner_checkpoint_sha256,
            geometry_refiner_checkpoint_contract)
        self.CLASSES = getattr(self.baseline, 'CLASSES', ('grab',))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._frozen_hash_at_init = self.frozen_parameter_hash()
        self._frozen_refiner_hash_at_init = self.frozen_refiner_hash()
        self._public_init_completed = False

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_refiner_checkpoint(self, checkpoint, expected_sha256,
                                 expected_contract=None):
        if checkpoint is None:
            if expected_sha256 is not None:
                raise ValueError(
                    'Refiner SHA256 was provided without a checkpoint')
            return dict(initialized_from_checkpoint=False)
        path = os.path.abspath(os.fspath(checkpoint))
        if not os.path.isfile(path):
            raise RuntimeError(
                'Geometry-refiner checkpoint does not exist: ' + path)
        observed = self._file_sha256(path)
        if (expected_sha256 is not None and
                observed.lower() != str(expected_sha256).lower()):
            raise RuntimeError(
                'Geometry-refiner checkpoint SHA256 mismatch')
        payload = torch.load(path, map_location='cpu')
        state = dict(payload.get('state_dict') or payload)
        local = {}
        prefixes = ('module.geometry_refiner.', 'geometry_refiner.')
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    local[key[len(prefix):]] = value
                    break
        if not local:
            raise RuntimeError(
                'Geometry-refiner checkpoint contains no refiner state')
        self.geometry_refiner.load_state_dict(local, strict=True)
        contract = dict(payload.get('meta') or {}).get(
            'geometry_refiner_checkpoint_contract')
        if not isinstance(contract, dict):
            raise RuntimeError(
                'Geometry-refiner checkpoint has no evidence contract')
        required = dict(
            architecture=self.geometry_refiner.component_contract().get(
                'architecture'),
            source_train_frames=self.evidence_contract[
                'source_train_frames'],
            source_val_frames=738,
            target_data_read=False,
            fixed_test_read=False,
            source_gate_passed=False,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False)
        required.update(dict(expected_contract or {}))
        failures = [
            '{}={!r}'.format(key, contract.get(key))
            for key, expected in required.items()
            if contract.get(key) != expected]
        if failures:
            raise RuntimeError(
                'Geometry-refiner initialization contract failed: ' +
                ', '.join(failures))
        return dict(
            initialized_from_checkpoint=True,
            path=path,
            sha256=observed,
            contract=contract)

    def init_weights(self):
        """Honor MMDetection's public init lifecycle without reinitializing.

        Both children are already in their final initial state here: the
        baseline was populated from the formal checkpoint and the refiner
        initialized itself (including its zero output layer) in ``__init__``.
        Calling ``BaseDetector.init_weights`` would recursively overwrite the
        frozen checkpoint before iteration zero.
        """
        current = self.frozen_parameter_hash()
        if current != self._frozen_hash_at_init:
            raise RuntimeError(
                'Frozen SymEOOD changed before public init_weights')
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.baseline.eval()
        if self.frozen_refiner_hash() != self._frozen_refiner_hash_at_init:
            raise RuntimeError(
                'Frozen refiner components changed before public init')
        self._public_init_completed = True

    def train(self, mode=True):
        super().train(mode)
        # MMCV runners call model.train() every epoch.  Keep BN/running state
        # and every frozen SymEOOD module in inference mode unconditionally.
        self.baseline.eval()
        self.geometry_refiner.train(mode)
        return self

    @staticmethod
    def _source_frame_identity(meta):
        filename = meta.get('ori_filename', meta.get('filename', ''))
        stem = os.path.splitext(os.path.basename(os.fspath(filename)))[0]
        match = re.match(r'^(real|sim)_(.+)_(\d+)$', stem)
        if match is None:
            return None
        return match.group(1), match.group(2), int(match.group(3))

    def _temporal_pair_indices(self, img_metas, proposal_list):
        rows = []
        offset = 0
        for meta, proposals in zip(img_metas, proposal_list):
            count = int(proposals.shape[0])
            if count == 1:
                identity = self._source_frame_identity(meta)
                if identity is not None:
                    rows.append((identity, offset))
            offset += count
        pairs = []
        for (first, first_row), (second, second_row) in zip(rows, rows[1:]):
            if (first[:2] == second[:2] and
                    second[2] == first[2] + 1):
                pairs.append((first_row, second_row))
        return pairs

    def extract_feat(self, img):
        with torch.no_grad():
            return tuple(feature.detach()
                         for feature in self.baseline.extract_feat(img))

    def extract_causal_history_feat(self, history_images):
        if history_images.ndim != 5:
            raise RuntimeError(
                'Causal history images must have shape [B,K,C,H,W]')
        batch, horizon = history_images.shape[:2]
        flattened = history_images.reshape(
            batch * horizon, *history_images.shape[2:])
        features = self.extract_feat(flattened)
        return tuple(feature.reshape(
            batch, horizon, *feature.shape[1:]) for feature in features)

    @staticmethod
    def _top_single_class_proposal(result, device, dtype):
        """Convert an MMRotate bbox result into its highest-score OBB."""
        if not isinstance(result, (list, tuple)):
            raise RuntimeError('Frozen K1 result must be a class-wise list')
        candidates = []
        for class_result in result:
            tensor = torch.as_tensor(
                class_result, device=device, dtype=dtype)
            if tensor.numel() == 0:
                continue
            if tensor.ndim != 2 or tensor.shape[1] < 6:
                raise RuntimeError('Frozen K1 result has invalid shape')
            candidates.append(tensor[:, :6])
        if not candidates:
            return torch.zeros((0, 5), device=device, dtype=dtype)
        candidates = torch.cat(candidates, dim=0)
        best = int(torch.argmax(candidates[:, 5]).item())
        return candidates[best:best + 1, :5]

    def _k1_results_and_proposals(self, features, img_metas, rescale=False):
        """Reuse frozen FPN features for K1 output and geometry anchors."""
        with torch.no_grad():
            results = self.baseline.simple_test_from_features(
                features, img_metas, rescale=rescale)
        proposals = [self._top_single_class_proposal(
            result, features[0].device, features[0].dtype)
            for result in results]
        return results, proposals

    def _causal_forward(self,
                        features,
                        proposal_list,
                        history_images,
                        history_proposals,
                        history_valid_mask,
                        history_ages,
                        conditioning_proposal_list=None):
        history_images = history_images.to(features[0].device)
        history_proposals = history_proposals.to(features[0].device)
        history_valid_mask = history_valid_mask.to(features[0].device)
        history_ages = history_ages.to(features[0].device)
        history_features = self.extract_causal_history_feat(history_images)
        kwargs = dict(
            history_features=history_features,
            history_proposals=history_proposals,
            history_valid_mask=history_valid_mask,
            history_ages=history_ages)
        if conditioning_proposal_list is not None:
            kwargs['conditioning_proposal_list'] = (
                conditioning_proposal_list)
        return self.geometry_refiner(features, proposal_list, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      dino_proposals,
                      causal_history_images=None,
                      causal_history_proposals=None,
                      causal_history_valid_mask=None,
                      causal_history_ages=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        del gt_labels, gt_bboxes_ignore, kwargs
        if self.evaluation_only:
            raise RuntimeError(
                'Evaluation-only geometry refiner cannot be trained')
        super().forward_train(img, img_metas)
        if len(dino_proposals) != len(gt_bboxes):
            raise RuntimeError('DINO/GT batch-size mismatch')
        features = self.extract_feat(img)
        dino_proposal_list = [item[:, :5].to(features[0].device)
                              for item in dino_proposals]
        if (self.uses_k1_geometry_anchor
                or self.uses_symmetric_dual_candidate_anchor):
            _, k1_proposal_list = self._k1_results_and_proposals(
                features, img_metas, rescale=False)
            if self.uses_symmetric_dual_candidate_anchor:
                proposal_list, conditioning_proposal_list = (
                    self.geometry_refiner.compose_symmetric_anchor(
                        k1_proposal_list, dino_proposal_list))
            else:
                proposal_list = [
                    k1 if int(k1.shape[0]) == 1 else dino
                    for k1, dino in zip(
                        k1_proposal_list, dino_proposal_list)]
                conditioning_proposal_list = dino_proposal_list
        else:
            proposal_list = dino_proposal_list
            conditioning_proposal_list = None
        gt_box_list = [item[:, :5].to(features[0].device)
                       for item in gt_bboxes]
        causal = hasattr(self.geometry_refiner, 'forward_causal')
        if causal:
            causal_inputs = (
                causal_history_images, causal_history_proposals,
                causal_history_valid_mask, causal_history_ages)
            if any(item is None for item in causal_inputs):
                raise RuntimeError(
                    'Causal refiner training requires complete history input')
            active = [
                index for index, proposal in enumerate(proposal_list)
                if (int(proposal.shape[0]) == 1
                    and (conditioning_proposal_list is None or int(
                        conditioning_proposal_list[index].shape[0]) == 1))]
            if any(int(proposal.shape[0]) > 1 for proposal in proposal_list):
                raise RuntimeError('Causal refiner accepts at most one proposal')
            features = tuple(feature[active] for feature in features)
            active_img_metas = [img_metas[index] for index in active]
            proposal_list = [proposal_list[index] for index in active]
            gt_box_list = [gt_box_list[index] for index in active]
            if conditioning_proposal_list is not None:
                conditioning_proposal_list = [
                    conditioning_proposal_list[index] for index in active]
            predicted = self._causal_forward(
                features, proposal_list,
                causal_history_images[active],
                causal_history_proposals[active],
                causal_history_valid_mask[active],
                causal_history_ages[active],
                conditioning_proposal_list=conditioning_proposal_list)
        else:
            predicted = self.geometry_refiner(features, proposal_list)
            active_img_metas = img_metas
        targets = self.geometry_refiner.encode_targets(
            proposal_list, gt_box_list)
        temporal_pairs = self._temporal_pair_indices(
            active_img_metas, proposal_list)
        losses = self.geometry_refiner.loss(
            predicted,
            targets,
            proposal_list=proposal_list,
            gt_box_list=gt_box_list,
            temporal_pair_indices=temporal_pairs)
        losses['refiner_sample_count'] = predicted.new_tensor(
            float(predicted.shape[0]))
        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    dino_proposals,
                    causal_history_images=None,
                    causal_history_proposals=None,
                    causal_history_valid_mask=None,
                    causal_history_ages=None,
                    rescale=False,
                    **kwargs):
        del kwargs
        features = self.extract_feat(img)
        dino_proposals = _unwrap_single_augmentation_proposals(
            dino_proposals)
        if len(dino_proposals) != len(img_metas):
            raise RuntimeError('DINO proposal/meta batch-size mismatch')
        dino_proposal_list = [item[:, :5].to(features[0].device)
                              for item in dino_proposals]
        if (self.uses_k1_geometry_anchor
                or self.uses_symmetric_dual_candidate_anchor):
            k1_results, k1_proposal_list = self._k1_results_and_proposals(
                features, img_metas, rescale=False)
            if self.uses_symmetric_dual_candidate_anchor:
                proposal_list, conditioning_proposal_list = (
                    self.geometry_refiner.compose_symmetric_anchor(
                        k1_proposal_list, dino_proposal_list))
            else:
                proposal_list = [
                    k1 if int(k1.shape[0]) == 1 else dino
                    for k1, dino in zip(
                        k1_proposal_list, dino_proposal_list)]
                conditioning_proposal_list = dino_proposal_list
        else:
            k1_results = None
            proposal_list = dino_proposal_list
            conditioning_proposal_list = dino_proposal_list
        causal = hasattr(self.geometry_refiner, 'forward_causal')
        if causal:
            causal_history_images = _unwrap_single_augmentation_tensor(
                causal_history_images, 'causal_history_images')
            causal_history_proposals = _unwrap_single_augmentation_tensor(
                causal_history_proposals, 'causal_history_proposals')
            causal_history_valid_mask = _unwrap_single_augmentation_tensor(
                causal_history_valid_mask, 'causal_history_valid_mask')
            causal_history_ages = _unwrap_single_augmentation_tensor(
                causal_history_ages, 'causal_history_ages')
            history_features = self.extract_causal_history_feat(
                causal_history_images.to(features[0].device))
        decoded = [None] * len(proposal_list)
        for index, proposals in enumerate(proposal_list):
            if (self.uses_k1_geometry_anchor
                    and int(dino_proposal_list[index].shape[0]) == 0):
                # Formal fallback: no DINO means the frozen K1 output is
                # returned exactly, without a history-only correction.
                continue
            if int(proposals.shape[0]) == 0:
                continue
            if int(proposals.shape[0]) != 1:
                raise RuntimeError(
                    'Geometry refiner accepts at most one proposal/image')
            image_features = tuple(
                feature[index:index + 1] for feature in features)
            if causal:
                image_history_features = tuple(
                    feature[index:index + 1] for feature in history_features)
                predicted = self.geometry_refiner(
                    image_features, [proposals],
                    conditioning_proposal_list=[
                        conditioning_proposal_list[index]],
                    history_features=image_history_features,
                    history_proposals=causal_history_proposals[
                        index:index + 1].to(features[0].device),
                    history_valid_mask=causal_history_valid_mask[
                        index:index + 1].to(features[0].device),
                    history_ages=causal_history_ages[
                        index:index + 1].to(features[0].device))
            else:
                predicted = self.geometry_refiner(
                    image_features, [proposals])
            decoded[index] = self.geometry_refiner.decode_and_normalize(
                [proposals], predicted, img_metas=[img_metas[index]])[0]
        outputs = []
        for index, boxes in enumerate(decoded):
            if boxes is None or boxes.shape[0] == 0:
                if k1_results is not None:
                    if rescale:
                        image_features = tuple(
                            feature[index:index + 1]
                            for feature in features)
                        fallback = self.baseline.simple_test_from_features(
                            image_features, [img_metas[index]],
                            rescale=True)[0]
                        outputs.append(fallback)
                    else:
                        outputs.append(k1_results[index])
                    continue
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

    def frozen_refiner_hash(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.geometry_refiner.state_dict().items()):
            parameter = dict(
                self.geometry_refiner.named_parameters()).get(name)
            if parameter is not None and parameter.requires_grad:
                continue
            digest.update(name.encode('utf-8'))
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(np.asarray(array).tobytes())
        return digest.hexdigest()

    def verify_frozen_contract(self):
        current = self.frozen_parameter_hash()
        frozen_refiner_current = self.frozen_refiner_hash()
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
            frozen_refiner_hash_at_init=self._frozen_refiner_hash_at_init,
            frozen_refiner_hash_current=frozen_refiner_current,
            frozen_refiner_hash_unchanged=(
                frozen_refiner_current == self._frozen_refiner_hash_at_init),
            public_init_completed=self._public_init_completed,
            evaluation_only=self.evaluation_only,
            geometry_refiner_initialization=dict(
                self.geometry_refiner_initialization),
            refiner_contract=self.geometry_refiner.component_contract(),
            evidence_contract=dict(self.evidence_contract))
