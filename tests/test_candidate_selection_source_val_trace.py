"""Check that the A/B report counts output and correctness separately."""

from crane_project.tools.trace_k1_candidate_selection_source_val_v1 import (
    summarize_rows)


def _arm(stage, output, good_score=None, bad_score=None):
    return dict(stage=stage, top1_index=0 if output else None,
                best_geometric_score=good_score,
                best_bad_score=bad_score,
                bad_minus_good_score=(bad_score - good_score
                                      if bad_score is not None
                                      and good_score is not None else None))


def test_trace_summary_separates_output_churn_from_riou_success():
    rows = [
        dict(frame='a_only', a=_arm('success', True, .8, .2),
             b=_arm('score_threshold', False, .03, .01)),
        dict(frame='b_only', a=_arm('score_threshold', False, .02, .01),
             b=_arm('success', True, .7, .1)),
        dict(frame='wrong_to_right', a=_arm('top1_ranking', True, .2, .4),
             b=_arm('success', True, .6, .3)),
    ]
    report = summarize_rows(rows)
    assert report['churn_counts'] == dict(a_only_output=1, b_only_output=1,
                                          a_only_success=1, b_only_success=2)
    assert report['stage_transitions']['top1_ranking'] == {'success': 1}
    assert report['paired_score_change']['best_geometric_score']['count'] == 3
    assert report['summaries']['a']['output_frames'] == 2
    assert report['summaries']['b']['output_frames'] == 2
