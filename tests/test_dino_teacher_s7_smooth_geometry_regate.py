from argparse import Namespace

import pytest

from crane_project.tools import (
    dino_teacher_s7_smooth_geometry_regate as regate)


def _summary(domains, sequences):
    metric = dict(net_top1_gain=1, top1_gains=1)
    return dict(
        native_wrong_s7_correct_pair_count=1,
        gain_domains=list(domains), gain_sequences=list(sequences),
        metrics={
            'sym_kld': dict(metric), 'gwd': dict(metric),
            'normalized_gwd': dict(metric)})


def _payload():
    return dict(
        protocol_version=27,
        protocol=dict(target_read=False),
        isolation=dict(parameter_updates_performed=False),
        source=dict(full=_summary(['real', 'sim'], ['real_seq07', 'sim_seq10']),
                    small=_summary(['sim'], ['sim_seq10'])),
        parameter_update_count=0, target_dev=None,
        candidate_forward_count=738)


def test_regate_is_offline_and_allows_current_small_coverage():
    args = Namespace(
        min_gain_domains=2, min_gain_sequences=2,
        small_min_gain_domains=1, small_min_gain_sequences=1)
    result = regate.build_regated_result(_payload(), args)
    assert result['protocol_version'] == 28
    assert result['eligible_for_training'] is True
    assert result['source']['support_gate']['coverage_limited'] is True
    assert result['new_model_forward_count'] == 0
    assert result['parameter_update_count'] == 0
    assert result['target_dev'] is None
    assert result['eligible_for_deployment'] is False


def test_regate_rejects_target_or_nonzero_update():
    args = Namespace(
        min_gain_domains=2, min_gain_sequences=2,
        small_min_gain_domains=1, small_min_gain_sequences=1)
    with pytest.raises(ValueError, match='target data'):
        regate.build_regated_result(dict(_payload(), target_dev={}), args)
    with pytest.raises(ValueError, match='parameter updates'):
        regate.build_regated_result(
            dict(_payload(), parameter_update_count=1), args)
