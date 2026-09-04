# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import hashlib
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes


def _validate_declared_checkpoint(cfg, checkpoint_path, checkpoint=None):
    """Enforce an optional config-level checkpoint identity contract.

    Normal evaluation configs do not declare these fields and retain the
    original tools/test.py behaviour.  Paired diagnostics can declare a
    canonical path, training protocol, and source-frame count so that an arm
    label cannot silently be evaluated with the other arm's weights.
    """
    expected_path = cfg.get('expected_checkpoint', None)
    if expected_path is not None:
        expected_realpath = osp.realpath(osp.abspath(osp.expanduser(
            os.fspath(expected_path))))
        observed_realpath = osp.realpath(osp.abspath(osp.expanduser(
            os.fspath(checkpoint_path))))
        if observed_realpath != expected_realpath:
            raise RuntimeError(
                'Checkpoint path does not match config contract: '
                f'observed={observed_realpath!r}, '
                f'expected={expected_realpath!r}')

    if checkpoint is None:
        return

    expected_protocol = cfg.get('expected_checkpoint_protocol', None)
    expected_frames = cfg.get(
        'expected_checkpoint_source_train_frames', None)
    expected_target_read = cfg.get(
        'expected_checkpoint_target_data_read', None)
    expected_fixed_test_read = cfg.get(
        'expected_checkpoint_fixed_test_read', None)
    if (expected_protocol is None and expected_frames is None
            and expected_target_read is None
            and expected_fixed_test_read is None):
        return
    contract = dict(checkpoint.get('meta') or {}).get(
        'geometry_refiner_checkpoint_contract')
    if not isinstance(contract, dict):
        raise RuntimeError(
            'Checkpoint has no geometry_refiner_checkpoint_contract')
    failures = []
    if (expected_protocol is not None
            and contract.get('protocol') != expected_protocol):
        failures.append(
            'protocol={!r} expected {!r}'.format(
                contract.get('protocol'), expected_protocol))
    if (expected_frames is not None
            and contract.get('source_train_frames') != expected_frames):
        failures.append(
            'source_train_frames={!r} expected {!r}'.format(
                contract.get('source_train_frames'), expected_frames))
    if (expected_target_read is not None
            and contract.get('target_data_read') is not expected_target_read):
        failures.append(
            'target_data_read={!r} expected {!r}'.format(
                contract.get('target_data_read'), expected_target_read))
    if (expected_fixed_test_read is not None and
            contract.get('fixed_test_read') is not expected_fixed_test_read):
        failures.append(
            'fixed_test_read={!r} expected {!r}'.format(
                contract.get('fixed_test_read'), expected_fixed_test_read))
    if failures:
        raise RuntimeError(
            'Checkpoint metadata does not match config contract: '
            + '; '.join(failures))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_eval_record(cfg, args, checkpoint, metric):
    """Build a provenance-rich evaluation record for paired diagnostics."""
    contract = dict(checkpoint.get('meta') or {}).get(
        'geometry_refiner_checkpoint_contract')
    return dict(
        config=osp.realpath(osp.abspath(args.config)),
        checkpoint=osp.realpath(osp.abspath(args.checkpoint)),
        checkpoint_sha256=_sha256_file(args.checkpoint),
        metric=metric,
        evidence_role=cfg.get('evidence_role', None),
        comparison_design=cfg.get('comparison_design', None),
        diagnostic_arm=cfg.get('diagnostic_arm', None),
        candidate_epoch_policy=cfg.get('candidate_epoch_policy', None),
        source_training_frame_count=cfg.get(
            'source_training_frame_count', None),
        auxiliary_source_frame_count=cfg.get(
            'auxiliary_source_frame_count', None),
        test_used_for_epoch_selection=cfg.get(
            'test_used_for_epoch_selection', None),
        eligible_for_unbiased_final_test_claim=cfg.get(
            'eligible_for_unbiased_final_test_claim', None),
        eligible_for_unknown_sequence_claim=cfg.get(
            'eligible_for_unknown_sequence_claim', None),
        checkpoint_contract=contract)


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--comparison-file',
        default='checkpoint_eval_summary.json',
        help='json file name under work-dir to append checkpoint metrics')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    _validate_declared_checkpoint(cfg, args.checkpoint)

    cfg = compat_cfg(cfg)

    if args.format_only and cfg.mp_start_method != 'spawn':
        warnings.warn(
            '`mp_start_method` in `cfg` is set to `spawn` to use CUDA '
            'with multiprocessing when formatting output result.')
        cfg.mp_start_method = 'spawn'

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if 'samples_per_gpu' in cfg.data.test:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            if 'samples_per_gpu' in ds_cfg:
                warnings.warn('`samples_per_gpu` in `test` field of '
                              'data will be deprecated, you should'
                              ' move it to `test_dataloader` field')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        test_dataloader_default_args['samples_per_gpu'] = samples_per_gpu
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    _validate_declared_checkpoint(cfg, args.checkpoint, checkpoint)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        raw_model = model.module if hasattr(model, 'module') else model
        audit_getter = getattr(raw_model, 'fusion_audit_records', None)
        if callable(audit_getter):
            audit_records = audit_getter()
            if audit_records:
                audit_dir = args.work_dir
                if audit_dir is None and args.out:
                    audit_dir = osp.dirname(osp.abspath(args.out))
                if audit_dir:
                    mmcv.mkdir_or_exist(audit_dir)
                    audit_name = cfg.get(
                        'fusion_audit_file', 'fusion_source_audit.json')
                    counts = {}
                    for record in audit_records:
                        source = record.get('output_source', 'unknown')
                        counts[source] = counts.get(source, 0) + 1
                    audit_path = osp.join(audit_dir, audit_name)
                    metadata_getter = getattr(
                        raw_model, 'fusion_audit_metadata', None)
                    metadata = (metadata_getter()
                                if callable(metadata_getter) else {})
                    protocol_getter = getattr(
                        raw_model, 'fusion_audit_protocol', None)
                    audit_protocol = (
                        protocol_getter() if callable(protocol_getter) else
                        'source_owned_geometry_union_v2')
                    mmcv.dump(dict(
                        protocol=audit_protocol,
                        frame_count=len(audit_records),
                        output_source_counts=counts,
                        metadata=metadata,
                        records=audit_records), audit_path)
                    print('writing fusion source audit to {}'.format(
                        audit_path))
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = _checkpoint_eval_record(
                cfg, args, checkpoint, metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)
                comparison_file = osp.join(args.work_dir, args.comparison_file)
                if osp.exists(comparison_file):
                    comparison = mmcv.load(comparison_file)
                    if not isinstance(comparison, list):
                        comparison = [comparison]
                else:
                    comparison = []
                comparison.append(metric_dict)
                mmcv.dump(comparison, comparison_file)


if __name__ == '__main__':
    main()
