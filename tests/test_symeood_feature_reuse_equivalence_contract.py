"""Pure report-contract checks for the source FPN reuse audit."""

from crane_project.tools.symeood_feature_reuse_equivalence import (
    _finalize_passed)


def test_integrated_runtime_failure_changes_final_status():
    report = dict(
        checks=dict(feature_equal=True),
        unified_runtime=dict(checks=dict(dino_once=False)))
    assert _finalize_passed(report) is False
    assert report['passed'] is False


def test_all_base_and_integrated_checks_are_required():
    report = dict(
        checks=dict(feature_equal=True, deterministic=True),
        unified_runtime=dict(
            checks=dict(dino_once=True, refiner_at_most_once=True)))
    assert _finalize_passed(report) is True
    assert report['passed'] is True


def test_base_only_mode_remains_supported():
    report = dict(checks=dict(feature_equal=True))
    assert _finalize_passed(report) is True
