"""The bounded supplement must stay in source train and avoid the first probe."""

from crane_project.tools.preflight_k1_candidate_selection_v1 import (
    choose_supplement_indices)


def test_supplement_is_deterministic_stratified_and_excludes_original_rows():
    first = choose_supplement_indices(2033, 748, 64)
    assert first == choose_supplement_indices(2033, 748, 64)
    assert len(first) == len(set(first)) == 128
    assert sum(index < 2033 for index in first) == 64
    assert sum(index >= 2033 for index in first) == 64
    assert {0, 1016, 2033, 2407}.isdisjoint(first)
