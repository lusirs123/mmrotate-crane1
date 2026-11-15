"""Regression checks for component eligibility, ties and denominator semantics."""
from crane_project.tools.run_current_component_reliability_v1 import (
    correct, eligible, risks, stats,
)
from crane_project.utils.port_simple_component_reliability_v1 import coverage_cutoff


def row(name, frame, pred, domain='sim'):
    return dict(image=name,sequence='seq',frame_id=frame,domain=domain,
                gt=[0,0,100,50,0],pred=pred,image_size=[1024,1024])


def test_missing_is_in_full_frame_coverage_not_false_rejection():
    rows=[row('a',1,[0,0,100,50,0,.9]),row('b',2,[0,0,120,50,0,.8]),
          row('c',3,None)]
    result=stats(rows,'size',{'a'})
    assert result['correct_accepted']==1
    assert result['incorrect_rejected']==1
    assert result['missing']==1
    assert result['all_frame_correct_coverage']==1/3
    assert result['longest_unaccepted_frames']==2


def test_sequence_and_frame_gaps_break_unavailability():
    rows=[row('a',1,None),row('b',3,None)]
    assert stats(rows,'size',set())['longest_unaccepted_frames']==1
    rows[1]['frame_id']=2
    rows[1]['sequence']='other'
    assert stats(rows,'size',set())['longest_unaccepted_frames']==1


def test_sim_only_angle_and_center_threshold():
    rows=[row('a',1,[15,0,100,50,0,.9]),row('b',2,None,'real')]
    assert len(eligible(rows,'angle'))==1
    assert len(eligible(rows,'size'))==2
    assert not correct(rows[0],'center')
    assert risks(rows,'center','simple',{})=={'a':0.}


def test_cutoff_accepts_entire_tie_without_using_labels():
    cutoff=coverage_cutoff([.1,.2,.2,.4],4,.5)
    assert cutoff['risk_le']==.2
    assert cutoff['actual_accepted']==3
